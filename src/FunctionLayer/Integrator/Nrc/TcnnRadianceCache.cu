#ifdef ENABLE_TCNN_NRC

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <mutex>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>

#include "FunctionLayer/Integrator/Nrc/TcnnRadianceCache.h"

namespace {

constexpr uint32_t kLegacyPlusInputDims = 18;
constexpr uint32_t kOutputDims = 3;
constexpr uint32_t kBatchSize = 1024;
constexpr float kDefaultRadianceScale = 1.0f;

using Precision = tcnn::network_precision_t;

struct PositionNormalizer {
    float center[3] = {0.0f, 0.0f, 0.0f};
    float extent[3] = {20.0f, 20.0f, 20.0f};
};

float clampFinite(double value, double minValue, double maxValue) {
    if (!std::isfinite(value)) {
        return 0.0f;
    }
    return static_cast<float>(std::max(minValue, std::min(maxValue, value)));
}

void encodeQuery(const RadianceQuery &query, const PositionNormalizer &positionNormalizer, float *out) {
    out[0] = clampFinite((query.position.x - positionNormalizer.center[0]) / positionNormalizer.extent[0] + 0.5, 0.0, 1.0);
    out[1] = clampFinite((query.position.y - positionNormalizer.center[1]) / positionNormalizer.extent[1] + 0.5, 0.0, 1.0);
    out[2] = clampFinite((query.position.z - positionNormalizer.center[2]) / positionNormalizer.extent[2] + 0.5, 0.0, 1.0);
    out[3] = clampFinite(query.normal.x * 0.5 + 0.5, 0.0, 1.0);
    out[4] = clampFinite(query.normal.y * 0.5 + 0.5, 0.0, 1.0);
    out[5] = clampFinite(query.normal.z * 0.5 + 0.5, 0.0, 1.0);
    out[6] = clampFinite(query.outgoing.x * 0.5 + 0.5, 0.0, 1.0);
    out[7] = clampFinite(query.outgoing.y * 0.5 + 0.5, 0.0, 1.0);
    out[8] = clampFinite(query.outgoing.z * 0.5 + 0.5, 0.0, 1.0);
    out[9] = clampFinite(query.albedo[0], 0.0, 1.0);
    out[10] = clampFinite(query.albedo[1], 0.0, 1.0);
    out[11] = clampFinite(query.albedo[2], 0.0, 1.0);
    out[12] = clampFinite(query.roughness, 0.0, 1.0);
    out[13] = clampFinite(query.materialType / 32.0, 0.0, 1.0);
    out[14] = clampFinite(query.bounce / 16.0, 0.0, 1.0);
    out[15] = clampFinite(query.specular[0], 0.0, 1.0);
    out[16] = clampFinite(query.specular[1], 0.0, 1.0);
    out[17] = clampFinite(query.specular[2], 0.0, 1.0);
}

void encodeTarget(const Spectrum &target, float scale, float *out) {
    scale = std::max(1.0e-4f, scale);
    out[0] = std::log1p(clampFinite(target[0], 0.0, 1.0e6) / scale);
    out[1] = std::log1p(clampFinite(target[1], 0.0, 1.0e6) / scale);
    out[2] = std::log1p(clampFinite(target[2], 0.0, 1.0e6) / scale);
}

float decodeRadiance(float value, float scale) {
    if (!std::isfinite(value)) {
        return 0.0f;
    }
    scale = std::max(1.0e-4f, scale);
    value = std::max(-20.0f, std::min(20.0f, value));
    return std::max(0.0f, std::expm1(value) * scale);
}

double luminanceOf(const Spectrum &value) {
    return 0.2126 * value[0] + 0.7152 * value[1] + 0.0722 * value[2];
}

Spectrum clampLuminancePreservingColor(const Spectrum &value, double maxLuminance) {
    if (maxLuminance <= 0.0) {
        return value;
    }
    const double luminance = luminanceOf(value);
    if (!std::isfinite(luminance) || luminance <= maxLuminance || luminance <= 1.0e-8) {
        return value;
    }
    return value * (maxLuminance / luminance);
}

double meanOf(const std::vector<double> &values) {
    if (values.empty()) {
        return 0.0;
    }
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

double percentile(std::vector<double> values, double q) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double index = std::max(0.0, std::min(1.0, q)) * static_cast<double>(values.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(index));
    const size_t hi = static_cast<size_t>(std::ceil(index));
    const double t = index - static_cast<double>(lo);
    return values[lo] * (1.0 - t) + values[hi] * t;
}

struct TcnnDiagnostics {
    size_t sampleCount = 0;
    size_t observedSampleCount = 0;
    int trainSteps = 0;
    int inputDims = kLegacyPlusInputDims;
    int relativeLoss = 0;
    int weightedSampleTraining = 0;
    double targetLuminanceClamp = 0.0;
    double trainingWeightMean = 1.0;
    double trainingWeightP95 = 1.0;
    double trainingWeightMax = 1.0;
    double initialLoss = 0.0;
    double finalLoss = 0.0;
    double targetMean = 0.0;
    double targetP50 = 0.0;
    double targetP95 = 0.0;
    double targetP99 = 0.0;
    double targetMax = 0.0;
    double radianceScale = kDefaultRadianceScale;
    double positionCenterX = 0.0;
    double positionCenterY = 0.0;
    double positionCenterZ = 0.0;
    double positionExtentX = 20.0;
    double positionExtentY = 20.0;
    double positionExtentZ = 20.0;
    double encodedTargetMean = 0.0;
    double encodedTargetP95 = 0.0;
    double validationEncodedMse = 0.0;
    double validationEncodedMae = 0.0;
    double validationDecodedMae = 0.0;
    double validationDecodedRelMae = 0.0;
    double validationTargetMean = 0.0;
    double validationPredMean = 0.0;
    double validationBiasR = 0.0;
    double validationBiasG = 0.0;
    double validationBiasB = 0.0;
    size_t coarseQueryBins = 0;
    size_t coarseConflictBins = 0;
    double coarseConflictSampleFraction = 0.0;
    double coarseTargetRangeMean = 0.0;
    double coarseTargetRangeP95 = 0.0;
    double coarseTargetRangeMax = 0.0;
    std::vector<std::pair<int, double>> lossCurve;
};

class TcnnRadianceCache final : public INeuralRadianceCache {
public:
    explicit TcnnRadianceCache(const NrcSettings &settings)
        : capacity(settings.maxTrainingSamples),
          inputDims(kLegacyPlusInputDims),
          useRelativeTarget(settings.tcnnRelativeTarget),
          weightedSampleTraining(settings.tcnnWeightedSampleTraining),
          trainingWeightClamp(settings.tcnnTrainingWeightClamp),
          targetLuminanceClamp(settings.targetLuminanceClamp),
          queryBatchSize(tcnn::next_multiple(1u, tcnn::BATCH_SIZE_GRANULARITY)),
          queryHostInput(queryBatchSize * inputDims, 0.0f),
          queryHostOutput(queryBatchSize * kOutputDims, 0.0f),
          queryInputMemory(queryBatchSize * inputDims),
          queryOutputMemory(queryBatchSize * kOutputDims) {
        cudaStreamCreate(&stream);

        const int hiddenLayers = std::max(1, settings.tcnnHiddenLayers);
        tcnn::json encodingConfig = {{"otype", "Frequency"}, {"n_frequencies", 4}};
        const std::string lossType = settings.tcnnRelativeTarget ? "RelativeL2" : "L2";
        tcnn::json config = {
            {"loss", {{"otype", lossType}}},
            {"optimizer", {
                {"otype", "Adam"},
                {"learning_rate", 1e-3},
                {"beta1", 0.9f},
                {"beta2", 0.99f},
                {"l2_reg", 1e-6f},
            }},
            {"encoding", encodingConfig},
            {"network", {
                {"otype", "FullyFusedMLP"},
                {"n_neurons", 64},
                {"n_hidden_layers", hiddenLayers},
                {"activation", "ReLU"},
                {"output_activation", "None"},
            }},
        };

        model = tcnn::create_from_config(inputDims, kOutputDims, config);
        model.network->set_jit_fusion(false);
    }

    ~TcnnRadianceCache() override {
        if (stream) {
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
        }
    }

    void reset() override {
        std::lock_guard<std::mutex> lock(mutex);
        samples.clear();
        trainCursor = 0;
        observedSampleCount = 0;
        trained = false;
        trainingSampleRng = 1;
        targetScale = kDefaultRadianceScale;
        positionNormalizer = {};
        diagnostics = {};
    }

    void enqueueTrainingSamples(const std::vector<RadianceSample> &newSamples) override {
        std::lock_guard<std::mutex> lock(mutex);
        for (const auto &sample : newSamples) {
            ++observedSampleCount;
            if (samples.size() < capacity) {
                samples.push_back(sample);
            } else {
                const size_t slot = reservoirSlot(observedSampleCount);
                if (slot < samples.size()) {
                    samples[slot] = sample;
                }
            }
        }
    }

    void train(int steps) override {
        std::lock_guard<std::mutex> lock(mutex);
        if (samples.empty() || steps <= 0) {
            return;
        }

        uint32_t batchSize = static_cast<uint32_t>(std::min(samples.size(), static_cast<size_t>(kBatchSize)));
        batchSize = tcnn::next_multiple(batchSize, tcnn::BATCH_SIZE_GRANULARITY);

        std::vector<float> hostInput(batchSize * inputDims, 0.0f);
        std::vector<float> hostTarget(batchSize * kOutputDims, 0.0f);
        tcnn::GPUMemory<float> inputMemory(hostInput.size());
        tcnn::GPUMemory<float> targetMemory(hostTarget.size());
        tcnn::GPUMatrix<float> input(inputMemory.data(), inputDims, batchSize);
        tcnn::GPUMatrix<float> target(targetMemory.data(), kOutputDims, batchSize);

        TcnnDiagnostics nextDiagnostics;
        nextDiagnostics.sampleCount = samples.size();
        nextDiagnostics.observedSampleCount = observedSampleCount;
        nextDiagnostics.trainSteps = steps;
        nextDiagnostics.inputDims = static_cast<int>(inputDims);
        nextDiagnostics.relativeLoss = useRelativeTarget ? 1 : 0;
        nextDiagnostics.weightedSampleTraining = weightedSampleTraining ? 1 : 0;
        nextDiagnostics.targetLuminanceClamp = targetLuminanceClamp;
        positionNormalizer = estimatePositionNormalizer();
        nextDiagnostics.positionCenterX = positionNormalizer.center[0];
        nextDiagnostics.positionCenterY = positionNormalizer.center[1];
        nextDiagnostics.positionCenterZ = positionNormalizer.center[2];
        nextDiagnostics.positionExtentX = positionNormalizer.extent[0];
        nextDiagnostics.positionExtentY = positionNormalizer.extent[1];
        nextDiagnostics.positionExtentZ = positionNormalizer.extent[2];
        targetScale = estimateTargetScale();
        nextDiagnostics.radianceScale = targetScale;
        summarizeTargets(nextDiagnostics);
        summarizeTrainingWeights(nextDiagnostics);
        summarizeQueryConflicts(nextDiagnostics);
        std::vector<double> cumulativeTrainingWeights;
        double totalTrainingWeight = 0.0;
        if (weightedSampleTraining) {
            cumulativeTrainingWeights = buildCumulativeTrainingWeights(totalTrainingWeight);
        }

        for (int i = 0; i < steps; ++i) {
            for (uint32_t j = 0; j < batchSize; ++j) {
                const size_t sampleIndex = weightedSampleTraining && totalTrainingWeight > 0.0
                    ? sampleWeightedTrainingIndex(cumulativeTrainingWeights, totalTrainingWeight)
                    : (trainCursor + j) % samples.size();
                const RadianceSample &sample = samples[sampleIndex];
                encodeQuery(sample.query, positionNormalizer, hostInput.data() + j * inputDims);
                encodeTarget(trainingRadiance(sample),
                             targetScale,
                             hostTarget.data() + j * kOutputDims);
            }
            trainCursor = (trainCursor + batchSize) % samples.size();
            inputMemory.copy_from_host(hostInput);
            targetMemory.copy_from_host(hostTarget);
            auto ctx = model.trainer->training_step(stream, input, target);
            if (i == 0 || i + 1 == steps || (i + 1) % 64 == 0) {
                const double loss = model.trainer->loss(stream, *ctx);
                nextDiagnostics.lossCurve.emplace_back(i + 1, loss);
                if (i == 0) {
                    nextDiagnostics.initialLoss = loss;
                }
                if (i + 1 == steps) {
                    nextDiagnostics.finalLoss = loss;
                }
            }
        }
        cudaStreamSynchronize(stream);
        trained = true;
        summarizeValidation(nextDiagnostics);
        diagnostics = std::move(nextDiagnostics);
    }

    Spectrum query(const RadianceQuery &query) const override {
        std::lock_guard<std::mutex> lock(mutex);
        if (!trained) {
            return Spectrum(0.0);
        }
        uint32_t batchSize = tcnn::next_multiple(1u, tcnn::BATCH_SIZE_GRANULARITY);
        std::fill(queryHostInput.begin(), queryHostInput.end(), 0.0f);
        encodeQuery(query, positionNormalizer, queryHostInput.data());
        queryInputMemory.copy_from_host(queryHostInput);

        tcnn::GPUMatrix<float> input(queryInputMemory.data(), inputDims, batchSize);
        tcnn::GPUMatrix<float> output(queryOutputMemory.data(), kOutputDims, batchSize);
        model.network->inference(stream, input, output);
        cudaStreamSynchronize(stream);

        queryOutputMemory.copy_to_host(queryHostOutput);
        Spectrum prediction(RGB3(
            decodeRadiance(static_cast<float>(queryHostOutput[0]), targetScale),
            decodeRadiance(static_cast<float>(queryHostOutput[1]), targetScale),
            decodeRadiance(static_cast<float>(queryHostOutput[2]), targetScale)));
        return prediction;
    }

    void queryBatch(const std::vector<RadianceQuery> &queries,
                    std::vector<Spectrum> &outputs) const override {
        std::lock_guard<std::mutex> lock(mutex);
        outputs.clear();
        outputs.resize(queries.size(), Spectrum(0.0));
        if (!trained || queries.empty()) {
            return;
        }

        uint32_t batchSize = static_cast<uint32_t>(queries.size());
        batchSize = tcnn::next_multiple(batchSize, tcnn::BATCH_SIZE_GRANULARITY);
        std::vector<float> hostInput(batchSize * inputDims, 0.0f);
        std::vector<float> hostOutput(batchSize * kOutputDims, 0.0f);
        for (size_t i = 0; i < queries.size(); ++i) {
            encodeQuery(queries[i], positionNormalizer, hostInput.data() + i * inputDims);
        }

        tcnn::GPUMemory<float> inputMemory(hostInput.size());
        tcnn::GPUMemory<float> outputMemory(hostOutput.size());
        inputMemory.copy_from_host(hostInput);

        tcnn::GPUMatrix<float> input(inputMemory.data(), inputDims, batchSize);
        tcnn::GPUMatrix<float> output(outputMemory.data(), kOutputDims, batchSize);
        model.network->inference(stream, input, output);
        cudaStreamSynchronize(stream);

        outputMemory.copy_to_host(hostOutput);
        for (size_t i = 0; i < queries.size(); ++i) {
            Spectrum prediction(RGB3(
                decodeRadiance(hostOutput[i * kOutputDims + 0], targetScale),
                decodeRadiance(hostOutput[i * kOutputDims + 1], targetScale),
                decodeRadiance(hostOutput[i * kOutputDims + 2], targetScale)));
            outputs[i] = prediction;
        }
    }

    bool writeDiagnostics(const std::string &path) const override {
        std::lock_guard<std::mutex> lock(mutex);
        std::ofstream file(path);
        if (!file) {
            return false;
        }

        file << "metric,value\n";
        file << "sample_count," << diagnostics.sampleCount << "\n";
        file << "observed_sample_count," << diagnostics.observedSampleCount << "\n";
        file << "train_steps," << diagnostics.trainSteps << "\n";
        file << "input_dims," << diagnostics.inputDims << "\n";
        file << "relative_loss," << diagnostics.relativeLoss << "\n";
        file << "weighted_sample_training," << diagnostics.weightedSampleTraining << "\n";
        file << "target_luminance_clamp," << diagnostics.targetLuminanceClamp << "\n";
        file << "training_weight_mean," << diagnostics.trainingWeightMean << "\n";
        file << "training_weight_p95," << diagnostics.trainingWeightP95 << "\n";
        file << "training_weight_max," << diagnostics.trainingWeightMax << "\n";
        file << "initial_loss," << diagnostics.initialLoss << "\n";
        file << "final_loss," << diagnostics.finalLoss << "\n";
        file << "target_luminance_mean," << diagnostics.targetMean << "\n";
        file << "target_luminance_p50," << diagnostics.targetP50 << "\n";
        file << "target_luminance_p95," << diagnostics.targetP95 << "\n";
        file << "target_luminance_p99," << diagnostics.targetP99 << "\n";
        file << "target_luminance_max," << diagnostics.targetMax << "\n";
        file << "radiance_scale," << diagnostics.radianceScale << "\n";
        file << "position_center_x," << diagnostics.positionCenterX << "\n";
        file << "position_center_y," << diagnostics.positionCenterY << "\n";
        file << "position_center_z," << diagnostics.positionCenterZ << "\n";
        file << "position_extent_x," << diagnostics.positionExtentX << "\n";
        file << "position_extent_y," << diagnostics.positionExtentY << "\n";
        file << "position_extent_z," << diagnostics.positionExtentZ << "\n";
        file << "encoded_target_mean," << diagnostics.encodedTargetMean << "\n";
        file << "encoded_target_p95," << diagnostics.encodedTargetP95 << "\n";
        file << "validation_encoded_mse," << diagnostics.validationEncodedMse << "\n";
        file << "validation_encoded_mae," << diagnostics.validationEncodedMae << "\n";
        file << "validation_decoded_mae," << diagnostics.validationDecodedMae << "\n";
        file << "validation_decoded_relative_mae," << diagnostics.validationDecodedRelMae << "\n";
        file << "validation_target_luminance_mean," << diagnostics.validationTargetMean << "\n";
        file << "validation_pred_luminance_mean," << diagnostics.validationPredMean << "\n";
        file << "validation_bias_r," << diagnostics.validationBiasR << "\n";
        file << "validation_bias_g," << diagnostics.validationBiasG << "\n";
        file << "validation_bias_b," << diagnostics.validationBiasB << "\n";
        file << "coarse_query_bins," << diagnostics.coarseQueryBins << "\n";
        file << "coarse_conflict_bins," << diagnostics.coarseConflictBins << "\n";
        file << "coarse_conflict_sample_fraction," << diagnostics.coarseConflictSampleFraction << "\n";
        file << "coarse_target_range_mean," << diagnostics.coarseTargetRangeMean << "\n";
        file << "coarse_target_range_p95," << diagnostics.coarseTargetRangeP95 << "\n";
        file << "coarse_target_range_max," << diagnostics.coarseTargetRangeMax << "\n";

        std::ofstream curve(path + ".loss.csv");
        if (curve) {
            curve << "step,loss\n";
            for (const auto &[step, loss] : diagnostics.lossCurve) {
                curve << step << "," << loss << "\n";
            }
        }
        return true;
    }

private:
    PositionNormalizer estimatePositionNormalizer() const {
        PositionNormalizer normalizer;
        if (samples.empty()) {
            return normalizer;
        }

        double minValue[3] = {
            samples[0].query.position.x,
            samples[0].query.position.y,
            samples[0].query.position.z,
        };
        double maxValue[3] = {
            samples[0].query.position.x,
            samples[0].query.position.y,
            samples[0].query.position.z,
        };
        for (const auto &sample : samples) {
            const double p[3] = {
                sample.query.position.x,
                sample.query.position.y,
                sample.query.position.z,
            };
            for (uint32_t c = 0; c < 3; ++c) {
                if (!std::isfinite(p[c])) {
                    continue;
                }
                minValue[c] = std::min(minValue[c], p[c]);
                maxValue[c] = std::max(maxValue[c], p[c]);
            }
        }
        for (uint32_t c = 0; c < 3; ++c) {
            const double extent = std::max(1.0e-3, (maxValue[c] - minValue[c]) * 1.05);
            normalizer.center[c] = static_cast<float>((minValue[c] + maxValue[c]) * 0.5);
            normalizer.extent[c] = static_cast<float>(extent);
        }
        return normalizer;
    }

    float estimateTargetScale() const {
        std::vector<double> channels;
        channels.reserve(samples.size() * kOutputDims);
        for (const auto &sample : samples) {
            const Spectrum target = trainingRadiance(sample);
            for (uint32_t c = 0; c < kOutputDims; ++c) {
                const double value = target[c];
                if (std::isfinite(value) && value > 0.0) {
                    channels.push_back(value);
                }
            }
        }
        if (channels.empty()) {
            return kDefaultRadianceScale;
        }
        const double p95 = percentile(channels, 0.95);
        return static_cast<float>(std::max(0.02, std::min(4.0, p95 * 0.5)));
    }

    void summarizeTargets(TcnnDiagnostics &out) const {
        std::vector<double> luminanceValues;
        std::vector<double> encodedValues;
        luminanceValues.reserve(samples.size());
        encodedValues.reserve(samples.size() * kOutputDims);
        float encoded[kOutputDims];
        for (const auto &sample : samples) {
            const Spectrum trainingTarget = trainingRadiance(sample);
            const double lum = luminanceOf(sample.target);
            if (std::isfinite(lum)) {
                luminanceValues.push_back(lum);
            }
            encodeTarget(trainingTarget, targetScale, encoded);
            for (uint32_t c = 0; c < kOutputDims; ++c) {
                encodedValues.push_back(encoded[c]);
            }
        }
        out.targetMean = meanOf(luminanceValues);
        out.targetP50 = percentile(luminanceValues, 0.50);
        out.targetP95 = percentile(luminanceValues, 0.95);
        out.targetP99 = percentile(luminanceValues, 0.99);
        out.targetMax = percentile(luminanceValues, 1.00);
        out.encodedTargetMean = meanOf(encodedValues);
        out.encodedTargetP95 = percentile(encodedValues, 0.95);
    }

    double clampedTrainingWeight(const RadianceSample &sample) const {
        if (!std::isfinite(sample.weight) || sample.weight <= 0.0) {
            return 1.0e-4;
        }
        const double maxWeight = std::max(1.0e-4, trainingWeightClamp);
        return std::min(sample.weight, maxWeight);
    }

    void summarizeTrainingWeights(TcnnDiagnostics &out) const {
        std::vector<double> weights;
        weights.reserve(samples.size());
        for (const auto &sample : samples) {
            weights.push_back(clampedTrainingWeight(sample));
        }
        out.trainingWeightMean = meanOf(weights);
        out.trainingWeightP95 = percentile(weights, 0.95);
        out.trainingWeightMax = percentile(weights, 1.0);
    }

    std::vector<double> buildCumulativeTrainingWeights(double &totalWeight) const {
        std::vector<double> cumulative;
        cumulative.reserve(samples.size());
        totalWeight = 0.0;
        for (const auto &sample : samples) {
            totalWeight += clampedTrainingWeight(sample);
            cumulative.push_back(totalWeight);
        }
        return cumulative;
    }

    double nextTrainingRandom01() {
        trainingSampleRng = trainingSampleRng * 2862933555777941757ULL + 3037000493ULL;
        const uint64_t bits = (trainingSampleRng >> 11) | 1ULL;
        return static_cast<double>(bits) * (1.0 / 9007199254740992.0);
    }

    size_t sampleWeightedTrainingIndex(const std::vector<double> &cumulative, double totalWeight) {
        if (cumulative.empty() || totalWeight <= 0.0) {
            return 0;
        }
        const double value = nextTrainingRandom01() * totalWeight;
        auto it = std::lower_bound(cumulative.begin(), cumulative.end(), value);
        if (it == cumulative.end()) {
            return cumulative.size() - 1;
        }
        return static_cast<size_t>(std::distance(cumulative.begin(), it));
    }

    size_t reservoirSlot(size_t observedCount) const {
        uint64_t x = static_cast<uint64_t>(observedCount);
        x ^= x >> 30;
        x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27;
        x *= 0x94d049bb133111ebULL;
        x ^= x >> 31;
        return observedCount == 0 ? 0 : static_cast<size_t>(x % observedCount);
    }

    void summarizeQueryConflicts(TcnnDiagnostics &out) const {
        struct BinStats {
            int count = 0;
            double minTarget = 1.0e30;
            double maxTarget = -1.0e30;
        };

        std::unordered_map<std::string, BinStats> bins;
        bins.reserve(samples.size());
        std::vector<float> encoded(inputDims, 0.0f);
        for (const auto &sample : samples) {
            encodeQuery(sample.query, positionNormalizer, encoded.data());
            std::string key;
            key.resize(inputDims);
            for (uint32_t d = 0; d < inputDims; ++d) {
                int q = static_cast<int>(std::round(clampFinite(encoded[d], -1.0, 1.0) * 15.5 + 15.5));
                q = std::max(0, std::min(31, q));
                key[d] = static_cast<char>(q);
            }
            const double target = luminanceOf(sample.target);
            auto &bin = bins[key];
            ++bin.count;
            bin.minTarget = std::min(bin.minTarget, target);
            bin.maxTarget = std::max(bin.maxTarget, target);
        }

        std::vector<double> ranges;
        ranges.reserve(bins.size());
        size_t conflictSamples = 0;
        for (const auto &[_, bin] : bins) {
            if (bin.count < 2) {
                continue;
            }
            const double range = bin.maxTarget - bin.minTarget;
            ranges.push_back(range);
            if (range > 0.05) {
                ++out.coarseConflictBins;
                conflictSamples += static_cast<size_t>(bin.count);
            }
        }
        out.coarseQueryBins = bins.size();
        out.coarseConflictSampleFraction = samples.empty()
            ? 0.0
            : static_cast<double>(conflictSamples) / static_cast<double>(samples.size());
        out.coarseTargetRangeMean = meanOf(ranges);
        out.coarseTargetRangeP95 = percentile(ranges, 0.95);
        out.coarseTargetRangeMax = percentile(ranges, 1.0);
    }

    void summarizeValidation(TcnnDiagnostics &out) const {
        if (!trained || samples.empty()) {
            return;
        }

        const uint32_t requested = static_cast<uint32_t>(std::min<size_t>(samples.size(), 4096));
        uint32_t batchSize = tcnn::next_multiple(requested, tcnn::BATCH_SIZE_GRANULARITY);
        std::vector<float> hostInput(batchSize * inputDims, 0.0f);
        std::vector<float> hostTarget(batchSize * kOutputDims, 0.0f);
        std::vector<float> hostOutput(batchSize * kOutputDims, 0.0f);

        for (uint32_t i = 0; i < requested; ++i) {
            const size_t index = samples.size() == requested
                ? static_cast<size_t>(i)
                : static_cast<size_t>((static_cast<uint64_t>(i) * samples.size()) / requested);
            const RadianceSample &sample = samples[index];
            encodeQuery(sample.query, positionNormalizer, hostInput.data() + i * inputDims);
            encodeTarget(trainingRadiance(sample),
                         targetScale,
                         hostTarget.data() + i * kOutputDims);
        }

        tcnn::GPUMemory<float> inputMemory(hostInput.size());
        tcnn::GPUMemory<float> outputMemory(hostOutput.size());
        inputMemory.copy_from_host(hostInput);

        tcnn::GPUMatrix<float> input(inputMemory.data(), inputDims, batchSize);
        tcnn::GPUMatrix<float> output(outputMemory.data(), kOutputDims, batchSize);
        model.network->inference(stream, input, output);
        cudaStreamSynchronize(stream);
        outputMemory.copy_to_host(hostOutput);

        double encMse = 0.0;
        double encMae = 0.0;
        double decMae = 0.0;
        double decRelMae = 0.0;
        double targetLum = 0.0;
        double predLum = 0.0;
        double bias[kOutputDims] = {0.0, 0.0, 0.0};

        for (uint32_t i = 0; i < requested; ++i) {
            const size_t index = samples.size() == requested
                ? static_cast<size_t>(i)
                : static_cast<size_t>((static_cast<uint64_t>(i) * samples.size()) / requested);
            const RadianceSample &sample = samples[index];
            Spectrum targetSpectrum = sample.target;
            Spectrum predTrainingSpectrum(0.0);
            for (uint32_t c = 0; c < kOutputDims; ++c) {
                const float predEncoded = hostOutput[i * kOutputDims + c];
                const float targetEncoded = hostTarget[i * kOutputDims + c];
                const double encodedDiff = static_cast<double>(predEncoded) - static_cast<double>(targetEncoded);
                encMse += encodedDiff * encodedDiff;
                encMae += std::abs(encodedDiff);

                const double predValue = decodeRadiance(predEncoded, targetScale);
                predTrainingSpectrum[c] = predValue;
            }
            Spectrum predSpectrum = predTrainingSpectrum;
            for (uint32_t c = 0; c < kOutputDims; ++c) {
                const double targetValue = targetSpectrum[c];
                const double predValue = predSpectrum[c];
                const double decodedDiff = predValue - targetValue;
                decMae += std::abs(decodedDiff);
                decRelMae += std::abs(decodedDiff) / (std::abs(targetValue) + 1.0);
                bias[c] += decodedDiff;
            }
            targetLum += luminanceOf(targetSpectrum);
            predLum += luminanceOf(predSpectrum);
        }

        const double count = static_cast<double>(requested);
        const double channelCount = count * static_cast<double>(kOutputDims);
        out.validationEncodedMse = encMse / channelCount;
        out.validationEncodedMae = encMae / channelCount;
        out.validationDecodedMae = decMae / channelCount;
        out.validationDecodedRelMae = decRelMae / channelCount;
        out.validationTargetMean = targetLum / count;
        out.validationPredMean = predLum / count;
        out.validationBiasR = bias[0] / count;
        out.validationBiasG = bias[1] / count;
        out.validationBiasB = bias[2] / count;
    }

    size_t capacity;
    uint32_t inputDims = kLegacyPlusInputDims;
    bool useRelativeTarget = false;
    bool weightedSampleTraining = false;
    double trainingWeightClamp = 8.0;
    double targetLuminanceClamp = 0.0;
    mutable std::mutex mutex;
    std::vector<RadianceSample> samples;
    size_t trainCursor = 0;
    size_t observedSampleCount = 0;
    tcnn::TrainableModel model;
    cudaStream_t stream = nullptr;
    bool trained = false;
    uint64_t trainingSampleRng = 1;
    float targetScale = kDefaultRadianceScale;
    PositionNormalizer positionNormalizer;
    uint32_t queryBatchSize = 0;
    mutable std::vector<float> queryHostInput;
    mutable std::vector<float> queryHostOutput;
    mutable tcnn::GPUMemory<float> queryInputMemory;
    mutable tcnn::GPUMemory<float> queryOutputMemory;
    TcnnDiagnostics diagnostics;

    Spectrum trainingRadiance(const RadianceSample &sample) const {
        return clampLuminancePreservingColor(sample.target, targetLuminanceClamp);
    }
};

} // namespace

std::shared_ptr<INeuralRadianceCache> CreateTcnnRadianceCache(const NrcSettings &settings) {
    return std::make_shared<TcnnRadianceCache>(settings);
}

#endif
