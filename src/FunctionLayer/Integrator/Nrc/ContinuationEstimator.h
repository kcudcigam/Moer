#pragma once

#include <memory>

#include "CoreLayer/ColorSpace/Color.h"
#include "FunctionLayer/Integrator/Nrc/INeuralRadianceCache.h"

class Scene;

struct ContinuationContext {
    Intersection intersection;
    Vec3d outgoing{0.0, 0.0, 1.0};
    Spectrum throughput{1.0};
    int bounce = 0;
    std::shared_ptr<Scene> scene;
};

class ContinuationEstimator {
public:
    virtual ~ContinuationEstimator() = default;

    virtual Spectrum estimate(const ContinuationContext &context) = 0;
};

class NrcContinuationEstimator : public ContinuationEstimator {
public:
    explicit NrcContinuationEstimator(std::shared_ptr<INeuralRadianceCache> cache)
        : radianceCache(std::move(cache)) {}

    Spectrum estimate(const ContinuationContext &context) override {
        RadianceQuery query = RadianceQuery::FromIntersection(
            context.intersection,
            context.outgoing,
            context.bounce);
        return radianceCache ? radianceCache->query(query) : Spectrum(0.0);
    }

private:
    std::shared_ptr<INeuralRadianceCache> radianceCache;
};
