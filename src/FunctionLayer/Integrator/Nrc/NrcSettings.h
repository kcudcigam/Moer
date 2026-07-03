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
    int trainingSpp = 0;
    int finalTrainSteps = 0;
    int minTrainingSamplesBeforeQuery = 0;
    int maxTrainingSamples = 1 << 16;
    double residualProbability = 0.125;
    double pathTraceBlend = 0.0;
    bool freezeAfterTraining = false;

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
        settings.trainingSpp = getOptional(json, "training_spp", settings.trainingSpp);
        settings.finalTrainSteps = getOptional(json, "final_train_steps", settings.finalTrainSteps);
        settings.minTrainingSamplesBeforeQuery = getOptional(json, "min_training_samples_before_query", settings.minTrainingSamplesBeforeQuery);
        settings.maxTrainingSamples = getOptional(json, "max_training_samples", settings.maxTrainingSamples);
        settings.residualProbability = getOptional(json, "residual_probability", settings.residualProbability);
        settings.pathTraceBlend = getOptional(json, "path_trace_blend", settings.pathTraceBlend);
        settings.freezeAfterTraining = getOptional(json, "freeze_after_training", settings.freezeAfterTraining);
        return settings;
    }
};
