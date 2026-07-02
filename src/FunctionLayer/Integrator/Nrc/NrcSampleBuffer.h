#pragma once

#include <deque>
#include <vector>

#include "FunctionLayer/Integrator/Nrc/RadianceQuery.h"

class NrcSampleBuffer {
public:
    explicit NrcSampleBuffer(size_t maxSamples = 1 << 16)
        : capacity(maxSamples) {}

    void push(const RadianceSample &sample) {
        if (samples.size() >= capacity) {
            samples.pop_front();
        }
        samples.push_back(sample);
    }

    void pushBatch(const std::vector<RadianceSample> &batch) {
        for (const auto &sample : batch) {
            push(sample);
        }
    }

    std::vector<RadianceSample> snapshot() const {
        return std::vector<RadianceSample>(samples.begin(), samples.end());
    }

    size_t size() const {
        return samples.size();
    }

    bool empty() const {
        return samples.empty();
    }

    void clear() {
        samples.clear();
    }

private:
    size_t capacity;
    std::deque<RadianceSample> samples;
};
