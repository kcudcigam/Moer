#include "FunctionLayer/Integrator/Nrc/RadianceCacheFactory.h"

#include <stdexcept>

#ifdef ENABLE_TCNN_NRC
#include "FunctionLayer/Integrator/Nrc/TcnnRadianceCache.h"
#endif

std::shared_ptr<INeuralRadianceCache> CreateRadianceCache(const NrcSettings &settings) {
#ifdef ENABLE_TCNN_NRC
    return CreateTcnnRadianceCache(settings);
#else
    (void)settings;
    throw std::runtime_error("TCNN NRC was requested, but ENABLE_TCNN_NRC is disabled.");
#endif
}
