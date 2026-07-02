#pragma once

#include <mutex>
#include <vector>

#include "FunctionLayer/Integrator/Nrc/INeuralRadianceCache.h"
#include "FunctionLayer/Integrator/Nrc/NrcSampleBuffer.h"

class RunningAverageRadianceCache : public INeuralRadianceCache {
public:
    explicit RunningAverageRadianceCache(size_t maxSamples = 1 << 16)
        : sampleBuffer(maxSamples) {}

    void reset() override {
        std::lock_guard<std::mutex> lock(mutex);
        sampleBuffer.clear();
        weightedSum = Spectrum(0.0);
        weightSum = 0.0;
        averageRadiance = Spectrum(0.0);
        hasAverage = false;
    }

    void enqueueTrainingSamples(const std::vector<RadianceSample> &samples) override {
        std::lock_guard<std::mutex> lock(mutex);
        sampleBuffer.pushBatch(samples);
        for (const auto &sample : samples) {
            double weight = sample.weight > 0.0 ? sample.weight : 1.0;
            weightedSum += sample.target * weight;
            weightSum += weight;
        }
    }

    void train(int) override {
        std::lock_guard<std::mutex> lock(mutex);
        if (weightSum <= 0.0) {
            return;
        }
        averageRadiance = weightedSum / weightSum;
        hasAverage = true;
    }

    Spectrum query(const RadianceQuery &) const override {
        std::lock_guard<std::mutex> lock(mutex);
        return hasAverage ? averageRadiance : Spectrum(0.0);
    }

    size_t sampleCount() const {
        std::lock_guard<std::mutex> lock(mutex);
        return sampleBuffer.size();
    }

private:
    NrcSampleBuffer sampleBuffer;
    Spectrum weightedSum{0.0};
    double weightSum = 0.0;
    Spectrum averageRadiance{0.0};
    bool hasAverage = false;
    mutable std::mutex mutex;
};
