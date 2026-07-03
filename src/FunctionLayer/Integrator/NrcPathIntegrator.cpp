#include "NrcPathIntegrator.h"

#include <chrono>
#include <cmath>
#include <limits>
#include <thread>

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
           bounce >= settings.queryBounce &&
           collectedTrainingSamples.load() >= settings.minTrainingSamplesBeforeQuery;
}

bool NrcPathIntegrator::shouldTrainCache() {
    if (settings.trainStepsPerRender <= 0) {
        return false;
    }
    const int batchSize = std::max(1, settings.trainBatchSize);
    const int sampleIndex = pendingTrainingSamples.fetch_add(1) + 1;
    return sampleIndex % batchSize == 0;
}

bool NrcPathIntegrator::isCacheableSurface(const Intersection &its) const {
    return its.material && its.material->type == EMaterialType::Diffuse;
}

bool NrcPathIntegrator::isTrustedCachedRadiance(const Spectrum &radiance) const {
    double maxChannel = 0.0;
    double minChannel = std::numeric_limits<double>::max();
    double sum = 0.0;
    for (int i = 0; i < 3; ++i) {
        double value = radiance[i];
        if (!std::isfinite(value) || value < 0.0) {
            return false;
        }
        maxChannel = std::max(maxChannel, value);
        minChannel = std::min(minChannel, value);
        sum += value;
    }
    if (maxChannel > 255.0 || sum > 512.0) {
        return false;
    }
    if (maxChannel > 4.0 && maxChannel > std::max(1.0, minChannel) * 8.0) {
        return false;
    }
    return true;
}

void NrcPathIntegrator::trainCache(int steps) {
    if (steps <= 0) {
        return;
    }
    auto trainStart = std::chrono::high_resolution_clock::now();
    radianceCache->train(steps);
    auto trainEnd = std::chrono::high_resolution_clock::now();
    stats.addTraining(std::chrono::duration<double>(trainEnd - trainStart).count());
}

Spectrum NrcPathIntegrator::traceContinuation(const Ray &ray, std::shared_ptr<Scene> scene) {
    return LiInternal(ray, scene, false, false, false);
}

Spectrum NrcPathIntegrator::Li(const Ray &initialRay, std::shared_ptr<Scene> scene) {
    return LiInternal(initialRay, scene, true, true, true);
}

void NrcPathIntegrator::renderTilePass(const std::shared_ptr<Scene> &scene,
                                       const std::vector<std::shared_ptr<Tile>> &tiles,
                                       int passSpp,
                                       bool depositToFilm,
                                       bool collectTraining,
                                       bool useCachedRadiance) {
    if (passSpp <= 0 || tiles.empty()) {
        return;
    }

    auto previousSampler = sampler;
    sampler = std::shared_ptr<Sampler>(previousSampler->clone(0));
    sampler->startPixel({0, 0});

    for (size_t tileIndex = 0; tileIndex < tiles.size(); ++tileIndex) {
        auto tile = tiles[tileIndex];
        for (auto it = tile->begin(); it != tile->end(); ++it) {
            auto pixelPosition = *it;
            sampler->startPixel(pixelPosition);
            for (int i = 0; i < passSpp; ++i) {
                auto ray = camera->generateRay(
                    film->getResolution(),
                    pixelPosition,
                    sampler->getCameraSample());
                auto L = LiInternal(ray, scene, true, collectTraining, useCachedRadiance);
                if (depositToFilm) {
                    film->deposit(pixelPosition, L);
                }
                sampler->nextSample();
            }
        }
        size_t done = tileIndex + 1;
        if (done % 5 == 0) {
            printProgress(static_cast<float>(done) / static_cast<float>(tiles.size()));
        }
    }
    sampler = previousSampler;
    printProgress(1.0f);
}

void NrcPathIntegrator::render(std::shared_ptr<Scene> scene) {
    if (settings.mode == NrcMode::PathTrace) {
        auto tiles = tileGenerator->generateTiles();
        renderTilePass(scene, tiles, spp, true, false, false);
        return;
    }

    if (!settings.freezeAfterTraining ||
        settings.trainingSpp <= 0) {
        MonteCarloIntegrator::render(scene);
        return;
    }

    auto tiles = tileGenerator->generateTiles();
    renderTilePass(scene, tiles, settings.trainingSpp, false, true, false);
    trainCache(settings.finalTrainSteps);
    pendingTrainingSamples.store(0);
    if (settings.mode == NrcMode::TwoLevel && residualCorrector) {
        renderTilePass(scene, tiles, settings.trainingSpp, false, true, true);
        pendingTrainingSamples.store(0);
    }
    renderTilePass(scene, tiles, spp, true, false, true);
}

Spectrum NrcPathIntegrator::LiInternal(const Ray &initialRay,
                                       std::shared_ptr<Scene> scene,
                                       bool enableEstimator,
                                       bool collectTrainingSamples,
                                       bool useCachedRadiance) {
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

        if (enableEstimator &&
            settings.mode != NrcMode::PathTrace &&
            continuationEstimator != nullptr &&
            isCacheableSurface(its) &&
            nBounces >= settings.queryBounce) {
            ContinuationContext context;
            context.intersection = its;
            context.outgoing = -ray.direction;
            context.throughput = throughput;
            context.bounce = nBounces;
            context.scene = scene;

            Spectrum targetRadiance(0.0);
            bool hasTargetRadiance = false;
            auto traceTargetRadiance = [&]() {
                if (hasTargetRadiance) {
                    return;
                }
                PathIntegratorLocalRecord sampleScatterRecord = sampleScatter(its, ray);
                if (sampleScatterRecord.f.isBlack() || sampleScatterRecord.pdf == 0) {
                    return;
                }
                Ray continuationRay{its.position + sampleScatterRecord.wi * eps, sampleScatterRecord.wi};
                Spectrum continuationThroughput = sampleScatterRecord.f / sampleScatterRecord.pdf;
                auto targetStart = std::chrono::high_resolution_clock::now();
                targetRadiance = continuationThroughput * traceContinuation(continuationRay, scene);
                auto targetEnd = std::chrono::high_resolution_clock::now();
                stats.addTargetTrace(std::chrono::duration<double>(targetEnd - targetStart).count());
                hasTargetRadiance = true;
            };

            if (collectTrainingSamples) {
                traceTargetRadiance();
            }

            if (hasTargetRadiance) {
                RadianceSample trainingSample;
                trainingSample.query = RadianceQuery::FromIntersection(its, -ray.direction, nBounces);
                trainingSample.target = targetRadiance;
                trainingSample.weight = 1.0;
                if (collectTrainingSamples) {
                    radianceCache->enqueueTrainingSamples({trainingSample});
                    collectedTrainingSamples.fetch_add(1);
                    stats.addTrainingSample();
                    if (shouldTrainCache()) {
                        trainCache(settings.trainStepsPerRender);
                    }
                }
            }

            Spectrum cachedRadiance(0.0);
            bool usedCachedRadiance = false;
            if (useCachedRadiance && shouldUseContinuationEstimator(nBounces)) {
                auto queryStart = std::chrono::high_resolution_clock::now();
                cachedRadiance = continuationEstimator->estimate(context);
                auto queryEnd = std::chrono::high_resolution_clock::now();
                stats.addQuery(std::chrono::duration<double>(queryEnd - queryStart).count());
                usedCachedRadiance = isTrustedCachedRadiance(cachedRadiance);
            }

            if (collectTrainingSamples && usedCachedRadiance && residualCorrector) {
                traceTargetRadiance();
                ResidualSample residualSample;
                residualSample.query = RadianceQuery::FromIntersection(its, -ray.direction, nBounces);
                residualSample.prediction = cachedRadiance;
                residualSample.pathTraceTarget = targetRadiance;
                residualSample.weight = 1.0;
                residualCorrector->enqueue(residualSample);
                stats.addResidualSample();
            }

            if (!collectTrainingSamples &&
                usedCachedRadiance &&
                settings.pathTraceBlend > 0.0) {
                traceTargetRadiance();
                if (hasTargetRadiance) {
                    double targetWeight = std::min(1.0, std::max(0.0, settings.pathTraceBlend));
                    cachedRadiance = cachedRadiance * (1.0 - targetWeight) + targetRadiance * targetWeight;
                }
            }

            if (!collectTrainingSamples &&
                usedCachedRadiance &&
                residualCorrector &&
                settings.residualProbability > 0.0 &&
                sampler->sample1D() < settings.residualProbability) {
                traceTargetRadiance();
                if (hasTargetRadiance) {
                    cachedRadiance += (targetRadiance - cachedRadiance) / settings.residualProbability;
                    stats.addResidualSample();
                }
            }

            if (usedCachedRadiance) {
                L += throughput * cachedRadiance;
            }
            if (usedCachedRadiance || collectTrainingSamples) {
                break;
            }
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
