#pragma once

#include <memory>

#include "FunctionLayer/Integrator/Nrc/INeuralRadianceCache.h"

std::shared_ptr<INeuralRadianceCache> CreateTcnnRadianceCache(size_t maxSamples);
