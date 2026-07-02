#pragma once

#include <string>

#include "CoreLayer/Adapter/JsonUtil.h"

enum class NrcMode {
    PathTrace,
    Nrc,
    TwoLevel
};

enum class NrcBackend {
    Cpu,
    Tcnn
};

struct NrcSettings {
    NrcMode mode = NrcMode::PathTrace;
    NrcBackend backend = NrcBackend::Cpu;
    int queryBounce = 2;
    int trainStepsPerRender = 0;
    int trainBatchSize = 1;
    int maxTrainingSamples = 1 << 16;
    double residualProbability = 0.125;

    static NrcSettings FromJson(const Json &json) {
        NrcSettings settings;
        std::string modeText = getOptional(json, "mode", std::string("path_trace"));
        if (modeText == "nrc") {
            settings.mode = NrcMode::Nrc;
        } else if (modeText == "two_level") {
            settings.mode = NrcMode::TwoLevel;
        } else {
            settings.mode = NrcMode::PathTrace;
        }
        std::string backendText = getOptional(json, "backend", std::string("cpu"));
        if (backendText == "tcnn") {
            settings.backend = NrcBackend::Tcnn;
        } else {
            settings.backend = NrcBackend::Cpu;
        }
        settings.queryBounce = getOptional(json, "query_bounce", settings.queryBounce);
        settings.trainStepsPerRender = getOptional(json, "train_steps", settings.trainStepsPerRender);
        settings.trainBatchSize = getOptional(json, "train_batch_size", settings.trainBatchSize);
        settings.maxTrainingSamples = getOptional(json, "max_training_samples", settings.maxTrainingSamples);
        settings.residualProbability = getOptional(json, "residual_probability", settings.residualProbability);
        return settings;
    }
};
