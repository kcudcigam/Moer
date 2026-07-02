#pragma once

#include <deque>
#include <mutex>
#include <vector>

#include "FunctionLayer/Integrator/Nrc/RadianceQuery.h"

struct ResidualSample {
    RadianceQuery query;
    Spectrum prediction{0.0};
    Spectrum pathTraceTarget{0.0};
    double weight = 1.0;

    Spectrum residual() const {
        return pathTraceTarget - prediction;
    }
};

class ResidualBuffer {
public:
    explicit ResidualBuffer(size_t maxSamples = 1 << 15)
        : capacity(maxSamples) {}

    void push(const ResidualSample &sample) {
        std::lock_guard<std::mutex> lock(mutex);
        if (samples.size() >= capacity) {
            samples.pop_front();
        }
        samples.push_back(sample);
    }

    std::vector<ResidualSample> snapshot() const {
        std::lock_guard<std::mutex> lock(mutex);
        return std::vector<ResidualSample>(samples.begin(), samples.end());
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex);
        return samples.size();
    }

private:
    size_t capacity;
    mutable std::mutex mutex;
    std::deque<ResidualSample> samples;
};

class ResidualCorrector {
public:
    explicit ResidualCorrector(size_t maxSamples = 1 << 15)
        : residualBuffer(maxSamples) {}

    void enqueue(const ResidualSample &sample) {
        residualBuffer.push(sample);
    }

    Spectrum estimateGlobalResidual() const {
        auto samples = residualBuffer.snapshot();
        if (samples.empty()) {
            return Spectrum(0.0);
        }

        Spectrum sum(0.0);
        double weightSum = 0.0;
        for (const auto &sample : samples) {
            double weight = sample.weight > 0.0 ? sample.weight : 1.0;
            sum += sample.residual() * weight;
            weightSum += weight;
        }
        return weightSum > 0.0 ? sum / weightSum : Spectrum(0.0);
    }

    size_t sampleCount() const {
        return residualBuffer.size();
    }

private:
    ResidualBuffer residualBuffer;
};
