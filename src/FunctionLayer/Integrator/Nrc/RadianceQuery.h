#pragma once

#include "CoreLayer/ColorSpace/Color.h"
#include "CoreLayer/Geometry/Geometry.h"
#include "FunctionLayer/Intersection.h"

struct RadianceQuery {
    Point3d position{0.0};
    Normal3d normal{0.0, 0.0, 1.0};
    Vec3d outgoing{0.0, 0.0, 1.0};
    Spectrum albedo{0.5};
    double roughness = 0.0;
    int materialType = 0;
    int bounce = 0;

    static RadianceQuery FromIntersection(const Intersection &its,
                                          const Vec3d &outgoing,
                                          int bounce) {
        RadianceQuery query;
        query.position = its.position;
        query.normal = its.geometryNormal;
        query.outgoing = outgoing;
        query.bounce = bounce;
        if (its.material) {
            query.materialType = static_cast<int>(its.material->type);
            auto bxdf = its.material->getBxDF(its);
            if (bxdf) {
                query.roughness = bxdf->getRoughness();
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
