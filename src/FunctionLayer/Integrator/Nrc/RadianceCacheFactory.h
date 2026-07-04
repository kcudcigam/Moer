#pragma once

#include <memory>

#include "FunctionLayer/Integrator/Nrc/INeuralRadianceCache.h"
#include "FunctionLayer/Integrator/Nrc/NrcSettings.h"

std::shared_ptr<INeuralRadianceCache> CreateRadianceCache(const NrcSettings &settings);
