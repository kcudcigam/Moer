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

Spectrum NrcPathIntegrator::traceContinuation(const Ray &ray,
                                              std::shared_ptr<Scene> scene,
                                              Sampler &localSampler) {
    return LiInternal(ray, scene, localSampler, false, false, false, false);
}

Spectrum NrcPathIntegrator::Li(const Ray &initialRay, std::shared_ptr<Scene> scene) {
    return LiInternal(initialRay, scene, *sampler, true, true, false, true);
}

PathIntegratorLocalRecord NrcPathIntegrator::sampleDirectLightingLocal(std::shared_ptr<Scene> scene,
                                                                       const Intersection &its,
                                                                       const Ray &ray,
                                                                       Sampler &localSampler) {
    auto [light, pdfChooseLight] = chooseOneLight(scene, localSampler.sample1D());
    auto record = light->sampleDirect(its, localSampler.sample2D(), ray.timeMin);
    double pdfDirect = record.pdfDirect * pdfChooseLight;
    Vec3d dirScatter = record.wi;
    Spectrum Li = record.s;
    Point3d posL = record.dst;
    Point3d posS = its.position;
    Spectrum transmittance(1.0);
    Ray visibilityTestingRay(posL - dirScatter * 1e-4, -dirScatter, ray.timeMin, ray.timeMax);
    auto visibilityTestingIts = scene->intersect(visibilityTestingRay);
    if (!visibilityTestingIts.has_value() ||
        visibilityTestingIts->object != its.object ||
        (visibilityTestingIts->position - posS).length2() > 1e-6) {
        transmittance = 0.0;
    }
    if (!visibilityTestingIts.has_value() && light->lightType == ELightType::INFINITE) {
        transmittance = 1.0;
    }
    return {dirScatter, Li * transmittance, pdfDirect, record.isDeltaPos};
}

PathIntegratorLocalRecord NrcPathIntegrator::sampleScatterLocal(const Intersection &its,
                                                                const Ray &ray,
                                                                Sampler &localSampler) {
    if (its.material == nullptr) {
        return {};
    }
    Vec3d wo = its.toLocal(-ray.direction);
    std::shared_ptr<BxDF> bxdf = its.material->getBxDF(its);
    Vec3d n = its.geometryNormal;
    BxDFSampleResult bsdfSample = bxdf->sample(wo, localSampler.sample2D(), false);
    double pdf = bsdfSample.pdf;
    Vec3d dirScatter = its.toWorld(bsdfSample.directionIn);
    double wiDotN = fm::abs(dot(dirScatter, n));
    return {dirScatter, bsdfSample.s * wiDotN, pdf, BxDF::MatchFlags(bsdfSample.bxdfSampleType, BXDF_SPECULAR)};
}

void NrcPathIntegrator::renderTilePass(const std::shared_ptr<Scene> &scene,
                                       const std::vector<std::shared_ptr<Tile>> &tiles,
                                       int passSpp,
                                       bool depositToFilm,
                                       bool collectTraining,
                                       bool collectResidual,
                                       bool useCachedRadiance) {
    if (passSpp <= 0 || tiles.empty()) {
        return;
    }

    std::atomic<size_t> nextTile{0};
    std::atomic<size_t> finishedTiles{0};
    std::vector<std::thread> threads;
    threads.reserve(renderThreadNum);

    for (int threadId = 0; threadId < renderThreadNum; ++threadId) {
        threads.emplace_back([&, threadId]() {
            auto localSampler = sampler->clone(threadId);
            localSampler->startPixel({0, 0});

            while (true) {
                size_t tileIndex = nextTile.fetch_add(1);
                if (tileIndex >= tiles.size()) {
                    break;
                }
                auto tile = tiles[tileIndex];
                for (auto it = tile->begin(); it != tile->end(); ++it) {
                    auto pixelPosition = *it;
                    localSampler->startPixel(pixelPosition);
                    for (int i = 0; i < passSpp; ++i) {
                        auto ray = camera->generateRay(
                            film->getResolution(),
                            pixelPosition,
                            localSampler->getCameraSample());
                        auto L = LiInternal(ray,
                                            scene,
                                            *localSampler,
                                            true,
                                            collectTraining,
                                            collectResidual,
                                            useCachedRadiance);
                        if (depositToFilm) {
                            film->deposit(pixelPosition, L);
                        }
                        localSampler->nextSample();
                    }
                }
                size_t done = finishedTiles.fetch_add(1) + 1;
                if (done % 5 == 0) {
                    printProgress(static_cast<float>(done) / static_cast<float>(tiles.size()));
                }
            }
        });
    }

    for (auto &thread : threads) {
        thread.join();
    }
    printProgress(1.0f);
}

void NrcPathIntegrator::render(std::shared_ptr<Scene> scene) {
    if (settings.mode == NrcMode::PathTrace) {
        auto tiles = tileGenerator->generateTiles();
        renderTilePass(scene, tiles, spp, true, false, false, false);
        return;
    }

    if (!settings.freezeAfterTraining ||
        settings.trainingSpp <= 0) {
        MonteCarloIntegrator::render(scene);
        return;
    }

    auto tiles = tileGenerator->generateTiles();
    renderTilePass(scene, tiles, settings.trainingSpp, false, true, false, false);
    trainCache(settings.finalTrainSteps);
    pendingTrainingSamples.store(0);
    if (settings.mode == NrcMode::TwoLevel && residualCorrector) {
        renderTilePass(scene, tiles, settings.trainingSpp, false, false, true, true);
        pendingTrainingSamples.store(0);
    }
    renderTilePass(scene, tiles, spp, true, false, false, true);
}

Spectrum NrcPathIntegrator::LiInternal(const Ray &initialRay,
                                       std::shared_ptr<Scene> scene,
                                       Sampler &localSampler,
                                       bool enableEstimator,
                                       bool collectTrainingSamples,
                                       bool collectResidualSamples,
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
        if (localSampler.sample1D() >= pSurvive) {
            break;
        }
        throughput /= pSurvive;

        for (int i = 0; i < nDirectLightSamples; ++i) {
            PathIntegratorLocalRecord sampleLightRecord = sampleDirectLightingLocal(scene, its, ray, localSampler);
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
                PathIntegratorLocalRecord sampleScatterRecord = sampleScatterLocal(its, ray, localSampler);
                if (sampleScatterRecord.f.isBlack() || sampleScatterRecord.pdf == 0) {
                    return;
                }
                Ray continuationRay{its.position + sampleScatterRecord.wi * eps, sampleScatterRecord.wi};
                Spectrum continuationThroughput = sampleScatterRecord.f / sampleScatterRecord.pdf;
                auto targetStart = std::chrono::high_resolution_clock::now();
                targetRadiance = continuationThroughput * traceContinuation(continuationRay, scene, localSampler);
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

            bool selectedResidualSample = false;
            if (collectResidualSamples) {
                if (!residualCorrector || settings.residualProbability <= 0.0) {
                    break;
                }
                selectedResidualSample = localSampler.sample1D() < settings.residualProbability;
                if (!selectedResidualSample) {
                    break;
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

            if (collectResidualSamples &&
                selectedResidualSample &&
                usedCachedRadiance &&
                residualCorrector) {
                traceTargetRadiance();
                if (hasTargetRadiance) {
                    ResidualSample residualSample;
                    residualSample.query = RadianceQuery::FromIntersection(its, -ray.direction, nBounces);
                    residualSample.prediction = cachedRadiance;
                    residualSample.pathTraceTarget = targetRadiance;
                    residualSample.weight = 1.0;
                    residualCorrector->enqueue(residualSample);
                    stats.addResidualSample();
                }
            }

            if (!collectTrainingSamples &&
                !collectResidualSamples &&
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
                localSampler.sample1D() < settings.residualProbability) {
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

        PathIntegratorLocalRecord sampleScatterRecord = sampleScatterLocal(its, ray, localSampler);
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
