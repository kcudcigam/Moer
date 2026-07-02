#pragma once

#include <vector>

#include "FunctionLayer/Integrator/Nrc/INeuralRadianceCache.h"

class DummyRadianceCache : public INeuralRadianceCache {
public:
    explicit DummyRadianceCache(Spectrum constantRadiance = Spectrum(0.0))
        : radiance(constantRadiance) {}

    void enqueueTrainingSamples(const std::vector<RadianceSample> &samples) override {
        sampleCount += samples.size();
    }

    void train(int steps) override {
        trainStepCount += steps;
    }

    Spectrum query(const RadianceQuery &) const override {
        return radiance;
    }

    size_t queuedSampleCount() const {
        return sampleCount;
    }

    int trainedSteps() const {
        return trainStepCount;
    }

private:
    Spectrum radiance;
    size_t sampleCount = 0;
    int trainStepCount = 0;
};
