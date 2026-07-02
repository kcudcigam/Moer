#include "NrcPathIntegrator.h"

#include <chrono>

#include "FastMath.h"
#include "FunctionLayer/Integrator/Nrc/RadianceCacheFactory.h"
#include "FunctionLayer/Material/NullMaterial.h"

NrcPathIntegrator::NrcPathIntegrator(std::shared_ptr<Camera> _camera,
                                     std::unique_ptr<Film> _film,
                                     std::unique_ptr<TileGenerator> _tileGenerator,
                                     std::shared_ptr<Sampler> _sampler,
                                     int _spp,
                                     const NrcSettings &_settings,
                                     int _renderThreadNum)
    : PathIntegratorNew(_camera,
                        std::move(_film),
                        std::move(_tileGenerator),
                        _sampler,
                        _spp,
                        _renderThreadNum),
      settings(_settings),
      radianceCache(CreateRadianceCache(_settings)) {
    if (settings.mode == NrcMode::Nrc) {
        continuationEstimator = std::make_unique<NrcContinuationEstimator>(radianceCache);
    } else if (settings.mode == NrcMode::TwoLevel) {
        residualCorrector = std::make_shared<ResidualCorrector>();
        continuationEstimator = std::make_unique<TwoLevelContinuationEstimator>(radianceCache, residualCorrector);
    }
}

bool NrcPathIntegrator::shouldUseContinuationEstimator(int bounce) const {
    return settings.mode != NrcMode::PathTrace &&
           continuationEstimator != nullptr &&
           bounce >= settings.queryBounce;
}

bool NrcPathIntegrator::shouldTrainCache() {
    if (settings.trainStepsPerRender <= 0) {
        return false;
    }
    const int batchSize = std::max(1, settings.trainBatchSize);
    const int sampleIndex = pendingTrainingSamples.fetch_add(1) + 1;
    return sampleIndex % batchSize == 0;
}

Spectrum NrcPathIntegrator::traceContinuation(const Ray &ray, std::shared_ptr<Scene> scene) {
    return LiInternal(ray, scene, false, false);
}

Spectrum NrcPathIntegrator::Li(const Ray &initialRay, std::shared_ptr<Scene> scene) {
    return LiInternal(initialRay, scene, true, true);
}

Spectrum NrcPathIntegrator::LiInternal(const Ray &initialRay,
                                       std::shared_ptr<Scene> scene,
                                       bool enableEstimator,
                                       bool collectTrainingSamples) {
    const double eps = 1e-4;
    Spectrum L{0.0};
    Spectrum throughput{1.0};
    Ray ray = initialRay;
    int nBounces = 0;
    auto itsOpt = scene->intersect(ray);

    while (true) {
        if (nBounces == 0) {
            PathIntegratorLocalRecord evalLightRecord = evalEmittance(scene, itsOpt, ray);
            L += throughput * evalLightRecord.f;
        }

        if (!itsOpt.has_value()) {
            break;
        }

        auto its = itsOpt.value();

        nBounces++;

        if (!its.material) {
            break;
        }

        auto bxdf = its.material->getBxDF(its);
        if (bxdf->isNull()) {
            nBounces--;
            ray = Ray{its.position + ray.direction * eps, ray.direction};
            itsOpt = scene->intersect(ray);
            continue;
        }

        double pSurvive = russianRoulette(throughput, nBounces);
        if (sampler->sample1D() >= pSurvive) {
            break;
        }
        throughput /= pSurvive;

        for (int i = 0; i < nDirectLightSamples; ++i) {
            PathIntegratorLocalRecord sampleLightRecord = sampleDirectLighting(scene, its, ray);
            PathIntegratorLocalRecord evalScatterRecord = evalScatter(its, ray, sampleLightRecord.wi);

            if (!sampleLightRecord.f.isBlack()) {
                double misw = MISWeight(sampleLightRecord.pdf, evalScatterRecord.pdf);
                if (sampleLightRecord.isDelta) {
                    misw = 1.0;
                }
                L += throughput * sampleLightRecord.f * evalScatterRecord.f
                     / sampleLightRecord.pdf * misw
                     / nDirectLightSamples;
            }
        }

        if (enableEstimator && shouldUseContinuationEstimator(nBounces)) {
            ContinuationContext context;
            context.intersection = its;
            context.outgoing = -ray.direction;
            context.throughput = throughput;
            context.bounce = nBounces;
            context.scene = scene;

            PathIntegratorLocalRecord sampleScatterRecord = sampleScatter(its, ray);
            Spectrum targetRadiance(0.0);
            if (!sampleScatterRecord.f.isBlack() && sampleScatterRecord.pdf != 0) {
                Ray continuationRay{its.position + sampleScatterRecord.wi * eps, sampleScatterRecord.wi};
                Spectrum continuationThroughput = sampleScatterRecord.f / sampleScatterRecord.pdf;
                auto targetStart = std::chrono::high_resolution_clock::now();
                targetRadiance = continuationThroughput * traceContinuation(continuationRay, scene);
                auto targetEnd = std::chrono::high_resolution_clock::now();
                stats.addTargetTrace(std::chrono::duration<double>(targetEnd - targetStart).count());

                RadianceSample trainingSample;
                trainingSample.query = RadianceQuery::FromIntersection(its, -ray.direction, nBounces);
                trainingSample.target = targetRadiance;
                trainingSample.weight = 1.0;
                if (collectTrainingSamples) {
                    radianceCache->enqueueTrainingSamples({trainingSample});
                    stats.addTrainingSample();
                    if (shouldTrainCache()) {
                        auto trainStart = std::chrono::high_resolution_clock::now();
                        radianceCache->train(settings.trainStepsPerRender);
                        auto trainEnd = std::chrono::high_resolution_clock::now();
                        stats.addTraining(std::chrono::duration<double>(trainEnd - trainStart).count());
                    }
                }
            }

            auto queryStart = std::chrono::high_resolution_clock::now();
            Spectrum cachedRadiance = continuationEstimator->estimate(context);
            auto queryEnd = std::chrono::high_resolution_clock::now();
            stats.addQuery(std::chrono::duration<double>(queryEnd - queryStart).count());

            if (collectTrainingSamples && residualCorrector) {
                ResidualSample residualSample;
                residualSample.query = RadianceQuery::FromIntersection(its, -ray.direction, nBounces);
                residualSample.prediction = cachedRadiance;
                residualSample.pathTraceTarget = targetRadiance;
                residualSample.weight = 1.0;
                residualCorrector->enqueue(residualSample);
                stats.addResidualSample();
            }

            L += throughput * cachedRadiance;
            break;
        }

        PathIntegratorLocalRecord sampleScatterRecord = sampleScatter(its, ray);
        if (!sampleScatterRecord.f.isBlack() && sampleScatterRecord.pdf != 0) {
            throughput *= sampleScatterRecord.f / sampleScatterRecord.pdf;
        } else {
            break;
        }

        ray = Ray{its.position + sampleScatterRecord.wi * eps, sampleScatterRecord.wi};
        itsOpt = scene->intersect(ray);

        auto evalLightRecord = evalEmittance(scene, itsOpt, ray);
        if (!evalLightRecord.f.isBlack()) {
            double misw = MISWeight(sampleScatterRecord.pdf, evalLightRecord.pdf);
            if (sampleScatterRecord.isDelta) {
                misw = 1.0;
            }

            L += throughput * evalLightRecord.f * misw;
        }
    }

    return L;
}
