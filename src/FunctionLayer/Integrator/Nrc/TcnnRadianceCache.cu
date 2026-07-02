#ifdef ENABLE_TCNN_NRC

#include <algorithm>
#include <cmath>
#include <mutex>
#include <vector>

#include <cuda_runtime.h>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>

#include "FunctionLayer/Integrator/Nrc/TcnnRadianceCache.h"

namespace {

constexpr uint32_t kInputDims = 12;
constexpr uint32_t kOutputDims = 3;
constexpr uint32_t kBatchSize = 1024;

using Precision = tcnn::network_precision_t;

float clampFinite(double value, double minValue, double maxValue) {
    if (!std::isfinite(value)) {
        return 0.0f;
    }
    return static_cast<float>(std::max(minValue, std::min(maxValue, value)));
}

void encodeQuery(const RadianceQuery &query, float *out) {
    out[0] = clampFinite(query.position.x * 0.05 + 0.5, 0.0, 1.0);
    out[1] = clampFinite(query.position.y * 0.05 + 0.5, 0.0, 1.0);
    out[2] = clampFinite(query.position.z * 0.05 + 0.5, 0.0, 1.0);
    out[3] = clampFinite(query.normal.x * 0.5 + 0.5, 0.0, 1.0);
    out[4] = clampFinite(query.normal.y * 0.5 + 0.5, 0.0, 1.0);
    out[5] = clampFinite(query.normal.z * 0.5 + 0.5, 0.0, 1.0);
    out[6] = clampFinite(query.outgoing.x * 0.5 + 0.5, 0.0, 1.0);
    out[7] = clampFinite(query.outgoing.y * 0.5 + 0.5, 0.0, 1.0);
    out[8] = clampFinite(query.outgoing.z * 0.5 + 0.5, 0.0, 1.0);
    out[9] = clampFinite(query.roughness, 0.0, 1.0);
    out[10] = clampFinite(query.materialType / 32.0, 0.0, 1.0);
    out[11] = clampFinite(query.bounce / 16.0, 0.0, 1.0);
}

void encodeTarget(const Spectrum &target, float *out) {
    out[0] = clampFinite(target[0], 0.0, 1.0e6);
    out[1] = clampFinite(target[1], 0.0, 1.0e6);
    out[2] = clampFinite(target[2], 0.0, 1.0e6);
}

class TcnnRadianceCache final : public INeuralRadianceCache {
public:
    explicit TcnnRadianceCache(size_t maxSamples)
        : capacity(maxSamples),
          queryBatchSize(tcnn::next_multiple(1u, tcnn::BATCH_SIZE_GRANULARITY)),
          queryHostInput(queryBatchSize * kInputDims, 0.0f),
          queryHostOutput(queryBatchSize * kOutputDims, 0.0f),
          queryInputMemory(queryBatchSize * kInputDims),
          queryOutputMemory(queryBatchSize * kOutputDims) {
        cudaStreamCreate(&stream);

        tcnn::json config = {
            {"loss", {{"otype", "L2"}}},
            {"optimizer", {
                {"otype", "Adam"},
                {"learning_rate", 1e-3},
                {"beta1", 0.9f},
                {"beta2", 0.99f},
                {"l2_reg", 1e-6f},
            }},
            {"encoding", {
                {"otype", "Frequency"},
                {"n_frequencies", 4},
            }},
            {"network", {
                {"otype", "FullyFusedMLP"},
                {"n_neurons", 64},
                {"n_hidden_layers", 2},
                {"activation", "ReLU"},
                {"output_activation", "None"},
            }},
        };

        model = tcnn::create_from_config(kInputDims, kOutputDims, config);
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
        trained = false;
    }

    void enqueueTrainingSamples(const std::vector<RadianceSample> &newSamples) override {
        std::lock_guard<std::mutex> lock(mutex);
        for (const auto &sample : newSamples) {
            if (samples.size() >= capacity) {
                samples.erase(samples.begin());
            }
            samples.push_back(sample);
        }
    }

    void train(int steps) override {
        std::lock_guard<std::mutex> lock(mutex);
        if (samples.empty() || steps <= 0) {
            return;
        }

        uint32_t batchSize = static_cast<uint32_t>(std::min(samples.size(), static_cast<size_t>(kBatchSize)));
        batchSize = tcnn::next_multiple(batchSize, tcnn::BATCH_SIZE_GRANULARITY);

        std::vector<float> hostInput(batchSize * kInputDims, 0.0f);
        std::vector<float> hostTarget(batchSize * kOutputDims, 0.0f);
        for (uint32_t i = 0; i < batchSize; ++i) {
            const RadianceSample &sample = samples[i % samples.size()];
            encodeQuery(sample.query, hostInput.data() + i * kInputDims);
            encodeTarget(sample.target, hostTarget.data() + i * kOutputDims);
        }

        tcnn::GPUMemory<float> inputMemory(hostInput.size());
        tcnn::GPUMemory<float> targetMemory(hostTarget.size());
        inputMemory.copy_from_host(hostInput);
        targetMemory.copy_from_host(hostTarget);

        tcnn::GPUMatrix<float> input(inputMemory.data(), kInputDims, batchSize);
        tcnn::GPUMatrix<float> target(targetMemory.data(), kOutputDims, batchSize);
        for (int i = 0; i < steps; ++i) {
            model.trainer->training_step(stream, input, target);
        }
        cudaStreamSynchronize(stream);
        trained = true;
    }

    Spectrum query(const RadianceQuery &query) const override {
        std::lock_guard<std::mutex> lock(mutex);
        if (!trained) {
            return Spectrum(0.0);
        }

        uint32_t batchSize = tcnn::next_multiple(1u, tcnn::BATCH_SIZE_GRANULARITY);
        std::fill(queryHostInput.begin(), queryHostInput.end(), 0.0f);
        encodeQuery(query, queryHostInput.data());
        queryInputMemory.copy_from_host(queryHostInput);

        tcnn::GPUMatrix<float> input(queryInputMemory.data(), kInputDims, batchSize);
        tcnn::GPUMatrix<float> output(queryOutputMemory.data(), kOutputDims, batchSize);
        model.network->inference(stream, input, output);
        cudaStreamSynchronize(stream);

        queryOutputMemory.copy_to_host(queryHostOutput);
        return Spectrum(RGB3(
            std::max(0.0f, static_cast<float>(queryHostOutput[0])),
            std::max(0.0f, static_cast<float>(queryHostOutput[1])),
            std::max(0.0f, static_cast<float>(queryHostOutput[2]))));
    }

private:
    size_t capacity;
    mutable std::mutex mutex;
    std::vector<RadianceSample> samples;
    tcnn::TrainableModel model;
    cudaStream_t stream = nullptr;
    bool trained = false;
    uint32_t queryBatchSize = 0;
    mutable std::vector<float> queryHostInput;
    mutable std::vector<float> queryHostOutput;
    mutable tcnn::GPUMemory<float> queryInputMemory;
    mutable tcnn::GPUMemory<float> queryOutputMemory;
};

} // namespace

std::shared_ptr<INeuralRadianceCache> CreateTcnnRadianceCache(size_t maxSamples) {
    return std::make_shared<TcnnRadianceCache>(maxSamples);
}

#endif
