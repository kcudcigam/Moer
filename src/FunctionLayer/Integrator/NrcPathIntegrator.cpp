#include "NrcPathIntegrator.h"

#include "FastMath.h"
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
      radianceCache(std::make_shared<DummyRadianceCache>()) {
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

Spectrum NrcPathIntegrator::Li(const Ray &initialRay, std::shared_ptr<Scene> scene) {
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

        if (shouldUseContinuationEstimator(nBounces)) {
            ContinuationContext context;
            context.intersection = its;
            context.outgoing = -ray.direction;
            context.throughput = throughput;
            context.bounce = nBounces;
            context.scene = scene;
            L += throughput * continuationEstimator->estimate(context);
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
