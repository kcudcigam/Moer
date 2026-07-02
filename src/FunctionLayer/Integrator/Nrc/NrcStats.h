#pragma once

#include <fstream>
#include <mutex>
#include <string>

class NrcStats {
public:
    void addQuery(double seconds) {
        std::lock_guard<std::mutex> lock(mutex);
        ++queryCount;
        querySeconds += seconds;
    }

    void addTrainingSample() {
        std::lock_guard<std::mutex> lock(mutex);
        ++trainingSamples;
    }

    void addTraining(double seconds) {
        std::lock_guard<std::mutex> lock(mutex);
        ++trainCalls;
        trainSeconds += seconds;
    }

    void addResidualSample() {
        std::lock_guard<std::mutex> lock(mutex);
        ++residualSamples;
    }

    void addTargetTrace(double seconds) {
        std::lock_guard<std::mutex> lock(mutex);
        targetTraceSeconds += seconds;
    }

    void writeCsv(const std::string &path) const {
        std::lock_guard<std::mutex> lock(mutex);
        std::ofstream file(path);
        file << "query_count,training_samples,residual_samples,train_calls,nrc_query_seconds,nrc_train_seconds,target_trace_seconds\n";
        file << queryCount << ","
             << trainingSamples << ","
             << residualSamples << ","
             << trainCalls << ","
             << querySeconds << ","
             << trainSeconds << ","
             << targetTraceSeconds << "\n";
    }

private:
    mutable std::mutex mutex;
    long long queryCount = 0;
    long long trainingSamples = 0;
    long long residualSamples = 0;
    long long trainCalls = 0;
    double querySeconds = 0.0;
    double trainSeconds = 0.0;
    double targetTraceSeconds = 0.0;
};
