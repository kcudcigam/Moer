#pragma once

#include <string>
#include <vector>

#include "FunctionLayer/Integrator/Nrc/RadianceQuery.h"

class INeuralRadianceCache {
public:
    virtual ~INeuralRadianceCache() = default;

    virtual void reset() {}

    virtual void enqueueTrainingSamples(const std::vector<RadianceSample> &samples) = 0;

    virtual void train(int steps) = 0;

    virtual Spectrum query(const RadianceQuery &query) const = 0;

    virtual void queryBatch(const std::vector<RadianceQuery> &queries,
                            std::vector<Spectrum> &outputs) const {
        outputs.clear();
        outputs.reserve(queries.size());
        for (const auto &query : queries) {
            outputs.push_back(this->query(query));
        }
    }

    virtual bool save(const std::string &) const { return false; }

    virtual bool load(const std::string &) { return false; }
};
