#include "mt_tensor_model.hpp"

#include "mt_background_1d.hpp"
#include "mt_config.hpp"

#include <algorithm>

namespace gmes::mt
{

namespace
{

int FindCell(const std::vector<double>& edges, double value)
{
    if (edges.size() < 2U)
    {
        return 0;
    }
    const auto it = std::upper_bound(edges.begin(), edges.end(), value);
    int idx = static_cast<int>(std::distance(edges.begin(), it)) - 1;
    return std::clamp(idx, 0, static_cast<int>(edges.size()) - 2);
}

} // namespace

TensorSigmaCoefficient::TensorSigmaCoefficient(const MTInput& input,
                                               const std::vector<double>& x_edges,
                                               const std::vector<double>& y_edges,
                                               const std::vector<double>& z_edges)
    : input_(input),
      x_edges_(x_edges),
      y_edges_(y_edges),
      z_edges_(z_edges),
      background_rho_(GetBackgroundRho(input))
{
}

double TensorSigmaCoefficient::RhoAtCell(int ix, int iy, int iz) const
{
    const int npad_xy = input_.padding.enabled ? input_.padding.npad_xy : 0;
    const int core_ix = ix - npad_xy;
    const int core_iy = iy - npad_xy;
    const int core_iz = iz;

    const bool in_core =
        core_ix >= 0 && core_ix < input_.model.nx &&
        core_iy >= 0 && core_iy < input_.model.ny &&
        core_iz >= 0 && core_iz < input_.model.nz;

    if (!in_core)
    {
        return background_rho_;
    }

    return std::max(
        input_.model.rho_ohm_m[TensorIndex(core_ix, core_iy, core_iz,
                                           input_.model.ny, input_.model.nz)],
        kMinimumRho);
}

double TensorSigmaCoefficient::Eval(mfem::ElementTransformation& transform,
                                    const mfem::IntegrationPoint& ip)
{
    mfem::Vector x(3);
    transform.Transform(ip, x);

    const int ix = FindCell(x_edges_, x[0]);
    const int iy = FindCell(y_edges_, x[1]);
    const int iz = FindCell(z_edges_, x[2]);
    return 1.0 / RhoAtCell(ix, iy, iz);
}

FrequencyScaledSigmaCoefficient::FrequencyScaledSigmaCoefficient(
    double scale, mfem::Coefficient& sigma)
    : scale_(scale), sigma_(sigma)
{
}

double FrequencyScaledSigmaCoefficient::Eval(mfem::ElementTransformation& transform,
                                             const mfem::IntegrationPoint& ip)
{
    return scale_ * sigma_.Eval(transform, ip);
}

} // namespace gmes::mt
