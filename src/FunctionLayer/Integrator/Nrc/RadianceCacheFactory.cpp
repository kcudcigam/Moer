#include "FunctionLayer/Integrator/Nrc/RadianceCacheFactory.h"

#include "FunctionLayer/Integrator/Nrc/RunningAverageRadianceCache.h"

#ifdef ENABLE_TCNN_NRC
#include "FunctionLayer/Integrator/Nrc/TcnnRadianceCache.h"
#endif

std::shared_ptr<INeuralRadianceCache> CreateRadianceCache(const NrcSettings &settings) {
#ifdef ENABLE_TCNN_NRC
    if (settings.backend == NrcBackend::Tcnn) {
        return CreateTcnnRadianceCache(settings.maxTrainingSamples);
    }
#endif
    return std::make_shared<RunningAverageRadianceCache>(settings.maxTrainingSamples);
}
