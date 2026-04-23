#pragma once

#include "mt_types.hpp"

namespace gmes::mt
{

double EstimateBoundaryBackgroundRho(const TensorModel& model);
double GetBackgroundRho(const MTInput& input);

} // namespace gmes::mt
