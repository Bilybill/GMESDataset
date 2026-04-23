#pragma once

#include "mt_types.hpp"

namespace gmes::mt
{

void ValidateInput(const MTInput& input);
MTInput NormalizeInput(MTInput input);
ReceiverGrid MakeDefaultReceiverGrid(const TensorModel& model);

} // namespace gmes::mt
