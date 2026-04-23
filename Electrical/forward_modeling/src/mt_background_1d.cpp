#include "mt_background_1d.hpp"

#include "mt_config.hpp"

#include <algorithm>
#include <stdexcept>

namespace gmes::mt
{

double EstimateBoundaryBackgroundRho(const TensorModel& model)
{
    if (model.nx <= 0 || model.ny <= 0 || model.nz <= 0)
    {
        throw std::invalid_argument("Cannot estimate background rho from an empty model.");
    }

    double sum = 0.0;
    std::size_t count = 0;
    for (int ix = 0; ix < model.nx; ++ix)
    {
        for (int iy = 0; iy < model.ny; ++iy)
        {
            for (int iz = 0; iz < model.nz; ++iz)
            {
                const bool is_boundary =
                    ix == 0 || ix == model.nx - 1 ||
                    iy == 0 || iy == model.ny - 1 ||
                    iz == 0 || iz == model.nz - 1;
                if (!is_boundary)
                {
                    continue;
                }
                sum += model.rho_ohm_m[TensorIndex(ix, iy, iz, model.ny, model.nz)];
                ++count;
            }
        }
    }

    return std::max(sum / static_cast<double>(count), kMinimumRho);
}

double GetBackgroundRho(const MTInput& input)
{
    return EstimateBoundaryBackgroundRho(input.model);
}

} // namespace gmes::mt
