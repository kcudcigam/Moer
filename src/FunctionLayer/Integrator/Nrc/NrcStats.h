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

    void addQueries(long long count, double seconds) {
        if (count <= 0) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex);
        queryCount += count;
        querySeconds += seconds;
    }

    void addTrainingSample() {
        std::lock_guard<std::mutex> lock(mutex);
        ++trainingSamples;
    }

    void addCacheableHit() {
        std::lock_guard<std::mutex> lock(mutex);
        ++cacheableHits;
    }

    void addEstimatorBlock() {
        std::lock_guard<std::mutex> lock(mutex);
        ++estimatorBlocks;
    }

    void addTraining(double seconds) {
        std::lock_guard<std::mutex> lock(mutex);
        ++trainCalls;
        trainSeconds += seconds;
    }

    void addTargetTrace(double seconds) {
        std::lock_guard<std::mutex> lock(mutex);
        targetTraceSeconds += seconds;
    }

    void writeCsv(const std::string &path) const {
        std::lock_guard<std::mutex> lock(mutex);
        std::ofstream file(path);
        file << "query_count,training_samples,train_calls,nrc_query_seconds,nrc_train_seconds,target_trace_seconds,cacheable_hits,estimator_blocks\n";
        file << queryCount << ","
             << trainingSamples << ","
             << trainCalls << ","
             << querySeconds << ","
             << trainSeconds << ","
             << targetTraceSeconds << ","
             << cacheableHits << ","
             << estimatorBlocks << "\n";
    }

private:
    mutable std::mutex mutex;
    long long queryCount = 0;
    long long trainingSamples = 0;
    long long trainCalls = 0;
    long long cacheableHits = 0;
    long long estimatorBlocks = 0;
    double querySeconds = 0.0;
    double trainSeconds = 0.0;
    double targetTraceSeconds = 0.0;
};
