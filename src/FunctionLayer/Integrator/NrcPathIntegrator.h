#pragma once

#include <memory>

#include "FunctionLayer/Integrator/Nrc/ContinuationEstimator.h"
#include "FunctionLayer/Integrator/Nrc/NrcSettings.h"
#include "FunctionLayer/Integrator/Nrc/RunningAverageRadianceCache.h"
#include "FunctionLayer/Integrator/PathIntegrator-new.h"

class NrcPathIntegrator : public PathIntegratorNew {
public:
    NrcPathIntegrator(std::shared_ptr<Camera> _camera,
                      std::unique_ptr<Film> _film,
                      std::unique_ptr<TileGenerator> _tileGenerator,
                      std::shared_ptr<Sampler> _sampler,
                      int _spp,
                      const NrcSettings &_settings,
                      int _renderThreadNum = 4);

    Spectrum Li(const Ray &initialRay, std::shared_ptr<Scene> scene) override;

    std::shared_ptr<INeuralRadianceCache> getRadianceCache() const {
        return radianceCache;
    }

private:
    NrcSettings settings;
    std::shared_ptr<INeuralRadianceCache> radianceCache;
    std::unique_ptr<ContinuationEstimator> continuationEstimator;

    bool shouldUseContinuationEstimator(int bounce) const;
    Spectrum traceContinuation(const Ray &ray, std::shared_ptr<Scene> scene);
    Spectrum LiInternal(const Ray &initialRay,
                        std::shared_ptr<Scene> scene,
                        bool enableEstimator,
                        bool collectTrainingSamples);
};
