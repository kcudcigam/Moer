#pragma once

#include <algorithm>

#include "CoreLayer/ColorSpace/Color.h"
#include "CoreLayer/Geometry/Geometry.h"
#include "FastMath.h"
#include "FunctionLayer/Intersection.h"

struct RadianceQuery {
    Point3d position{0.0};
    Normal3d normal{0.0, 0.0, 1.0};
    Vec3d outgoing{0.0, 0.0, 1.0};
    Point2d uv{0.0};
    Spectrum albedo{0.5};
    Spectrum specular{0.0};
    double roughness = 0.0;
    double viewCos = 1.0;
    int materialType = 0;
    int bounce = 0;

    static RadianceQuery FromIntersection(const Intersection &its,
                                          const Vec3d &outgoing,
                                          int bounce) {
        RadianceQuery query;
        query.position = its.position;
        query.normal = its.shFrame.n;
        query.outgoing = outgoing;
        query.uv = its.uv;
        query.bounce = bounce;
        if (its.material) {
            query.materialType = static_cast<int>(its.material->type);
            auto bxdf = its.material->getBxDF(its);
            if (bxdf) {
                query.roughness = bxdf->getRoughness();
                Vec3d woLocal = its.toLocal(outgoing);
                query.viewCos = std::max(0.0, std::min(1.0, woLocal.z));
                Spectrum reflectance = (bxdf->f(woLocal, Vec3d(0.0, 0.0, 1.0)) * fm::pi_d).clamp(0.0, 1.0);
                if (its.material->type == EMaterialType::Diffuse) {
                    query.albedo = reflectance;
                    query.specular = Spectrum(0.0);
                } else {
                    query.albedo = Spectrum(0.0);
                    query.specular = reflectance;
                }
            }
        }
        return query;
    }
};

struct RadianceSample {
    RadianceQuery query;
    Spectrum target{0.0};
    double weight = 1.0;
};
