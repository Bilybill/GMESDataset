#include "mt_driver.hpp"

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <utility>

namespace gmes::mt
{

void ValidateInput(const MTInput& input)
{
    const TensorModel& model = input.model;
    if (model.nx <= 0 || model.ny <= 0 || model.nz <= 0)
    {
        throw std::invalid_argument("MT model dimensions must be positive.");
    }
    if (model.dx <= 0.0 || model.dy <= 0.0 || model.dz <= 0.0)
    {
        throw std::invalid_argument("MT cell sizes must be positive.");
    }

    const std::size_t expected =
        static_cast<std::size_t>(model.nx) *
        static_cast<std::size_t>(model.ny) *
        static_cast<std::size_t>(model.nz);
    if (model.rho_ohm_m.size() != expected)
    {
        throw std::invalid_argument("rho_ohm_m size does not match model dimensions.");
    }

    for (double rho : model.rho_ohm_m)
    {
        if (rho <= 0.0)
        {
            throw std::invalid_argument("All resistivity values must be positive.");
        }
    }

    if (input.freqs_hz.empty())
    {
        throw std::invalid_argument("At least one MT frequency is required.");
    }
    for (double freq : input.freqs_hz)
    {
        if (freq <= 0.0)
        {
            throw std::invalid_argument("All MT frequencies must be positive.");
        }
    }

    if (input.solver.order <= 0)
    {
        throw std::invalid_argument("MFEM finite-element order must be positive.");
    }
}

ReceiverGrid MakeDefaultReceiverGrid(const TensorModel& model)
{
    ReceiverGrid receivers;
    receivers.x.reserve(static_cast<std::size_t>(model.nx));
    receivers.y.reserve(static_cast<std::size_t>(model.ny));
    for (int ix = 0; ix < model.nx; ++ix)
    {
        receivers.x.push_back((static_cast<double>(ix) + 0.5) * model.dx);
    }
    for (int iy = 0; iy < model.ny; ++iy)
    {
        receivers.y.push_back((static_cast<double>(iy) + 0.5) * model.dy);
    }
    receivers.z = 0.0;
    return receivers;
}

MTInput NormalizeInput(MTInput input)
{
    ValidateInput(input);
    std::sort(input.freqs_hz.begin(), input.freqs_hz.end(), std::greater<double>());

    if (input.receivers.x.empty() || input.receivers.y.empty())
    {
        input.receivers = MakeDefaultReceiverGrid(input.model);
    }
    return input;
}

} // namespace gmes::mt
