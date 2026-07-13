#pragma once

#include <string>

#include "CoreLayer/Adapter/JsonUtil.h"

enum class NrcMode {
    PathTrace,
    Tcnn
};

enum class NrcQuerySemantics {
    CurrentVertex,
    PostScatter
};

struct NrcSettings {
    NrcMode mode = NrcMode::PathTrace;
    NrcQuerySemantics querySemantics = NrcQuerySemantics::CurrentVertex;
    int queryBounce = 2;
    int trainStepsPerRender = 0;
    int trainBatchSize = 1;
    int trainingSpp = 0;
    int finalTrainSteps = 0;
    double finalTrainEpochs = 0.0;
    int minTrainingSamplesBeforeQuery = 0;
    int maxTrainingSamples = 1 << 16;
    double targetLuminanceClamp = 0.0;
    bool freezeAfterTraining = false;
    bool cacheNonDiffuseSurfaces = false;
    int targetSamples = 1;
    std::string tcnnEncoding = "legacy";
    int tcnnHiddenLayers = 2;
    bool tcnnRelativeTarget = false;
    bool tcnnWeightedSampleTraining = false;
    double tcnnTrainingWeightClamp = 8.0;
    bool useBatchQuery = true;
    bool countZeroTargetSamples = false;

    static NrcSettings FromJson(const Json &json) {
        NrcSettings settings;
        std::string modeText = getOptional(json, "mode", std::string("path_trace"));
        if (modeText == "nrc") {
            settings.mode = NrcMode::Tcnn;
        } else {
            settings.mode = NrcMode::PathTrace;
        }
        std::string querySemanticsText = getOptional(json, "query_semantics", std::string("current_vertex"));
        if (querySemanticsText == "post_scatter") {
            settings.querySemantics = NrcQuerySemantics::PostScatter;
        } else {
            settings.querySemantics = NrcQuerySemantics::CurrentVertex;
        }
        settings.queryBounce = getOptional(json, "query_bounce", settings.queryBounce);
        settings.trainStepsPerRender = getOptional(json, "train_steps", settings.trainStepsPerRender);
        settings.trainBatchSize = getOptional(json, "train_batch_size", settings.trainBatchSize);
        settings.trainingSpp = getOptional(json, "training_spp", settings.trainingSpp);
        settings.finalTrainSteps = getOptional(json, "final_train_steps", settings.finalTrainSteps);
        settings.finalTrainEpochs = getOptional(json, "final_train_epochs", settings.finalTrainEpochs);
        settings.minTrainingSamplesBeforeQuery = getOptional(json, "min_training_samples_before_query", settings.minTrainingSamplesBeforeQuery);
        settings.maxTrainingSamples = getOptional(json, "max_training_samples", settings.maxTrainingSamples);
        settings.targetLuminanceClamp = getOptional(json, "target_luminance_clamp", settings.targetLuminanceClamp);
        settings.freezeAfterTraining = getOptional(json, "freeze_after_training", settings.freezeAfterTraining);
        settings.cacheNonDiffuseSurfaces = getOptional(json, "cache_non_diffuse_surfaces", settings.cacheNonDiffuseSurfaces);
        settings.targetSamples = getOptional(json, "target_samples", settings.targetSamples);
        settings.tcnnEncoding = getOptional(json, "tcnn_encoding", settings.tcnnEncoding);
        settings.tcnnHiddenLayers = getOptional(json, "tcnn_hidden_layers", settings.tcnnHiddenLayers);
        settings.tcnnRelativeTarget = getOptional(json, "tcnn_relative_target", settings.tcnnRelativeTarget);
        settings.tcnnWeightedSampleTraining = getOptional(json, "tcnn_weighted_sample_training", settings.tcnnWeightedSampleTraining);
        settings.tcnnTrainingWeightClamp = getOptional(json, "tcnn_training_weight_clamp", settings.tcnnTrainingWeightClamp);
        settings.useBatchQuery = getOptional(json, "use_batch_query", settings.useBatchQuery);
        settings.countZeroTargetSamples = getOptional(json, "count_zero_target_samples", settings.countZeroTargetSamples);
        return settings;
    }

    int querySemanticsId() const {
        return querySemantics == NrcQuerySemantics::PostScatter ? 1 : 0;
    }
};
