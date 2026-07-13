#pragma once

#include <atomic>
#include <memory>
#include <vector>

#include "FunctionLayer/Integrator/Nrc/ContinuationEstimator.h"
#include "FunctionLayer/Integrator/Nrc/NrcSettings.h"
#include "FunctionLayer/Integrator/Nrc/NrcStats.h"
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
    void render(std::shared_ptr<Scene> scene) override;

    std::shared_ptr<INeuralRadianceCache> getRadianceCache() const {
        return radianceCache;
    }

    void writeStats(const std::string &path) const {
        stats.writeCsv(path);
        if (radianceCache) {
            radianceCache->writeDiagnostics(path + ".cache.csv");
        }
    }

private:
    struct BatchedSampleWork {
        Point2i pixel{0, 0};
        Spectrum radiance{0.0};
        Spectrum throughput{1.0};
        RadianceQuery query;
    };

    NrcSettings settings;
    std::shared_ptr<INeuralRadianceCache> radianceCache;
    std::unique_ptr<ContinuationEstimator> continuationEstimator;
    NrcStats stats;
    std::atomic<int> pendingTrainingSamples{0};
    std::atomic<long long> collectedTrainingSamples{0};

    bool shouldUseContinuationEstimator(int bounce) const;
    bool shouldQueryCache(int bounce, double pathSpread, double primarySpread) const;
    bool shouldTrainCache();
    bool isCacheableSurface(const Intersection &its) const;
    Spectrum sanitizeCachedRadiance(const Spectrum &radiance) const;
    void trainCache(int steps);
    void renderTilePass(const std::shared_ptr<Scene> &scene,
                        const std::vector<std::shared_ptr<Tile>> &tiles,
                        int passSpp,
                        bool depositToFilm,
                        bool collectTrainingSamples,
                        bool useCachedRadiance);
    void renderTilePassBatched(const std::shared_ptr<Scene> &scene,
                               const std::vector<std::shared_ptr<Tile>> &tiles,
                               int passSpp);
    bool traceToBatchedQuery(const Ray &initialRay,
                             std::shared_ptr<Scene> scene,
                             Sampler &localSampler,
                             const Point2i &pixel,
                             BatchedSampleWork &work);
    PathIntegratorLocalRecord sampleDirectLightingLocal(std::shared_ptr<Scene> scene,
                                                        const Intersection &its,
                                                        const Ray &ray,
                                                        Sampler &localSampler);
    PathIntegratorLocalRecord sampleScatterLocal(const Intersection &its,
                                                 const Ray &ray,
                                                 Sampler &localSampler);
    Spectrum traceContinuation(const Ray &ray, std::shared_ptr<Scene> scene, Sampler &localSampler);
    Spectrum LiInternal(const Ray &initialRay,
                        std::shared_ptr<Scene> scene,
                        Sampler &localSampler,
                        bool enableEstimator,
                        bool collectTrainingSamples,
                        bool useCachedRadiance);
};
