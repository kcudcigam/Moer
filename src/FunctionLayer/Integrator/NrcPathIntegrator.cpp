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
      radianceCache(_settings.mode == NrcMode::Tcnn ? CreateRadianceCache(_settings) : nullptr) {
    if (settings.mode == NrcMode::Tcnn) {
        continuationEstimator = std::make_unique<NrcContinuationEstimator>(radianceCache);
    }
}

bool NrcPathIntegrator::shouldUseContinuationEstimator(int bounce) const {
    return settings.mode != NrcMode::PathTrace &&
           continuationEstimator != nullptr &&
           bounce >= settings.queryBounce &&
           collectedTrainingSamples.load() >= settings.minTrainingSamplesBeforeQuery;
}

bool NrcPathIntegrator::shouldQueryCache(int bounce, double pathSpread, double primarySpread) const {
    (void)pathSpread;
    (void)primarySpread;
    if (bounce < settings.queryBounce) {
        return false;
    }
    return true;
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
    if (!its.material || its.material->type == EMaterialType::Null) {
        return false;
    }
    if (its.material->type == EMaterialType::Diffuse) {
        return true;
    }
    if (!settings.cacheNonDiffuseSurfaces) {
        return false;
    }
    auto bxdf = its.material->getBxDF(its);
    if (!bxdf || bxdf->isNull()) {
        return false;
    }
    return bxdf->getRoughness() > 1.0e-3;
}

Spectrum NrcPathIntegrator::sanitizeCachedRadiance(const Spectrum &radiance) const {
    Spectrum result = radiance;
    for (int i = 0; i < 3; ++i) {
        double value = radiance[i];
        if (!std::isfinite(value) || value < 0.0) {
            result[i] = 0.0;
        }
    }
    return result;
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
    return LiInternal(ray, scene, localSampler, false, false, false);
}

Spectrum NrcPathIntegrator::Li(const Ray &initialRay, std::shared_ptr<Scene> scene) {
    return LiInternal(initialRay, scene, *sampler, true, true, true);
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

bool NrcPathIntegrator::traceToBatchedQuery(const Ray &initialRay,
                                            std::shared_ptr<Scene> scene,
                                            Sampler &localSampler,
                                            const Point2i &pixel,
                                            BatchedSampleWork &work) {
    const double eps = 1e-4;
    Spectrum L{0.0};
    Spectrum throughput{1.0};
    Ray ray = initialRay;
    int nBounces = 0;
    double primarySpread = 0.0;
    double pathSpread = 0.0;
    double previousScatterPdf = 1.0;
    bool postScatterQueryReady = settings.querySemantics == NrcQuerySemantics::CurrentVertex;
    auto itsOpt = scene->intersect(ray);

    while (true) {
        if (nBounces == 0) {
            PathIntegratorLocalRecord evalLightRecord = evalEmittance(scene, itsOpt, ray);
            L += throughput * evalLightRecord.f;
        }

        if (!itsOpt.has_value()) {
            work.pixel = pixel;
            work.radiance = L;
            return false;
        }

        auto its = itsOpt.value();
        nBounces++;
        const double segmentLength2 = (its.position - ray.origin).length2();
        const double cosAtVertex = std::max(1.0e-4, fm::abs(dot(-ray.direction, its.geometryNormal)));
        if (nBounces == 1) {
            primarySpread = segmentLength2 / (4.0 * fm::pi_d * cosAtVertex);
        } else {
            const double denom = std::max(1.0e-4, previousScatterPdf * cosAtVertex);
            pathSpread += segmentLength2 / (denom * denom);
        }

        if (!its.material) {
            work.pixel = pixel;
            work.radiance = L;
            return false;
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
            work.pixel = pixel;
            work.radiance = L;
            return false;
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

        const bool cacheableSurface = isCacheableSurface(its);
        if (cacheableSurface) {
            stats.addCacheableHit();
        }
        if (settings.mode != NrcMode::PathTrace &&
            cacheableSurface &&
            postScatterQueryReady &&
            shouldUseContinuationEstimator(nBounces) &&
            shouldQueryCache(nBounces, pathSpread, primarySpread)) {
            stats.addEstimatorBlock();
            work.pixel = pixel;
            work.radiance = L;
            work.throughput = throughput;
            work.query = RadianceQuery::FromIntersection(its, -ray.direction, nBounces);
            return true;
        }

        PathIntegratorLocalRecord sampleScatterRecord = sampleScatterLocal(its, ray, localSampler);
        if (!sampleScatterRecord.f.isBlack() && sampleScatterRecord.pdf != 0) {
            throughput *= sampleScatterRecord.f / sampleScatterRecord.pdf;
            previousScatterPdf = sampleScatterRecord.pdf;
            if (settings.querySemantics == NrcQuerySemantics::PostScatter &&
                nBounces >= settings.queryBounce) {
                postScatterQueryReady = true;
            }
        } else {
            work.pixel = pixel;
            work.radiance = L;
            return false;
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
}

void NrcPathIntegrator::renderTilePassBatched(const std::shared_ptr<Scene> &scene,
                                              const std::vector<std::shared_ptr<Tile>> &tiles,
                                              int passSpp) {
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

            std::vector<BatchedSampleWork> pending;
            std::vector<RadianceQuery> queries;
            std::vector<Spectrum> cachedRadiance;

            while (true) {
                size_t tileIndex = nextTile.fetch_add(1);
                if (tileIndex >= tiles.size()) {
                    break;
                }

                pending.clear();
                queries.clear();

                auto tile = tiles[tileIndex];
                for (auto it = tile->begin(); it != tile->end(); ++it) {
                    auto pixelPosition = *it;
                    localSampler->startPixel(pixelPosition);
                    for (int i = 0; i < passSpp; ++i) {
                        auto ray = camera->generateRay(
                            film->getResolution(),
                            pixelPosition,
                            localSampler->getCameraSample());
                        BatchedSampleWork work;
                        if (traceToBatchedQuery(ray, scene, *localSampler, pixelPosition, work)) {
                            queries.push_back(work.query);
                            pending.push_back(work);
                        } else {
                            film->deposit(pixelPosition, work.radiance);
                        }
                        localSampler->nextSample();
                    }
                }

                if (!queries.empty()) {
                    auto queryStart = std::chrono::high_resolution_clock::now();
                    radianceCache->queryBatch(queries, cachedRadiance);
                    auto queryEnd = std::chrono::high_resolution_clock::now();
                    stats.addQueries(
                        static_cast<long long>(queries.size()),
                        std::chrono::duration<double>(queryEnd - queryStart).count());

                    for (size_t i = 0; i < pending.size(); ++i) {
                        Spectrum continuation = sanitizeCachedRadiance(cachedRadiance[i]);
                        film->deposit(pending[i].pixel, pending[i].radiance + pending[i].throughput * continuation);
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
    int finalTrainSteps = settings.finalTrainSteps;
    if (settings.finalTrainEpochs > 0.0) {
        constexpr int tcnnTrainBatchSize = 1024;
        const auto samples = static_cast<double>(
            std::min<long long>(collectedTrainingSamples.load(), settings.maxTrainingSamples));
        const int epochSteps = static_cast<int>(
            std::ceil(settings.finalTrainEpochs * samples / static_cast<double>(tcnnTrainBatchSize)));
        finalTrainSteps = std::max(finalTrainSteps, epochSteps);
    }
    trainCache(finalTrainSteps);
    pendingTrainingSamples.store(0);
    if (settings.useBatchQuery) {
        renderTilePassBatched(scene, tiles, spp);
    } else {
        renderTilePass(scene, tiles, spp, true, false, true);
    }
}

Spectrum NrcPathIntegrator::LiInternal(const Ray &initialRay,
                                       std::shared_ptr<Scene> scene,
                                       Sampler &localSampler,
                                       bool enableEstimator,
                                       bool collectTrainingSamples,
                                       bool useCachedRadiance) {
    const double eps = 1e-4;
    Spectrum L{0.0};
    Spectrum throughput{1.0};
    Ray ray = initialRay;
    int nBounces = 0;
    double primarySpread = 0.0;
    double pathSpread = 0.0;
    double previousScatterPdf = 1.0;
    bool postScatterQueryReady = settings.querySemantics == NrcQuerySemantics::CurrentVertex;
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
        const double segmentLength2 = (its.position - ray.origin).length2();
        const double cosAtVertex = std::max(1.0e-4, fm::abs(dot(-ray.direction, its.geometryNormal)));
        if (nBounces == 1) {
            primarySpread = segmentLength2 / (4.0 * fm::pi_d * cosAtVertex);
        } else {
            const double denom = std::max(1.0e-4, previousScatterPdf * cosAtVertex);
            pathSpread += segmentLength2 / (denom * denom);
        }

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

        const bool cacheableSurface = isCacheableSurface(its);
        if (cacheableSurface) {
            stats.addCacheableHit();
        }
        if (enableEstimator &&
            settings.mode != NrcMode::PathTrace &&
            continuationEstimator != nullptr &&
            cacheableSurface &&
            postScatterQueryReady &&
            shouldQueryCache(nBounces, pathSpread, primarySpread)) {
            stats.addEstimatorBlock();
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
                Spectrum accumulatedTarget(0.0);
                int validTargets = 0;
                auto targetStart = std::chrono::high_resolution_clock::now();
                const int targetSamples = std::max(1, settings.targetSamples);
                for (int targetIndex = 0; targetIndex < targetSamples; ++targetIndex) {
                    PathIntegratorLocalRecord sampleScatterRecord = sampleScatterLocal(its, ray, localSampler);
                    if (sampleScatterRecord.f.isBlack() || sampleScatterRecord.pdf == 0) {
                        if (settings.countZeroTargetSamples) {
                            ++validTargets;
                        }
                        continue;
                    }
                    Ray continuationRay{its.position + sampleScatterRecord.wi * eps, sampleScatterRecord.wi};
                    Spectrum continuationThroughput = sampleScatterRecord.f / sampleScatterRecord.pdf;
                    Spectrum continuation = traceContinuation(continuationRay, scene, localSampler);
                    accumulatedTarget += continuationThroughput * continuation;
                    ++validTargets;
                }
                auto targetEnd = std::chrono::high_resolution_clock::now();
                stats.addTargetTrace(std::chrono::duration<double>(targetEnd - targetStart).count());
                if (validTargets > 0) {
                    targetRadiance = accumulatedTarget / static_cast<double>(validTargets);
                    hasTargetRadiance = true;
                }
            };

            if (collectTrainingSamples) {
                traceTargetRadiance();
            }

            if (hasTargetRadiance) {
                RadianceSample trainingSample;
                trainingSample.query = RadianceQuery::FromIntersection(its, -ray.direction, nBounces);
                trainingSample.target = targetRadiance;
                if (settings.tcnnWeightedSampleTraining) {
                    trainingSample.weight = std::max(
                        1.0e-4,
                        0.2126 * throughput[0] + 0.7152 * throughput[1] + 0.0722 * throughput[2]);
                }
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
                cachedRadiance = sanitizeCachedRadiance(continuationEstimator->estimate(context));
                auto queryEnd = std::chrono::high_resolution_clock::now();
                stats.addQuery(std::chrono::duration<double>(queryEnd - queryStart).count());
                usedCachedRadiance = true;
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
            previousScatterPdf = sampleScatterRecord.pdf;
            if (settings.querySemantics == NrcQuerySemantics::PostScatter &&
                nBounces >= settings.queryBounce) {
                postScatterQueryReady = true;
            }
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
