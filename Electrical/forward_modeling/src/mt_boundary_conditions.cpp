#include "mt_boundary_conditions.hpp"

#include "mt_background_1d.hpp"

#include <algorithm>
#include <cmath>

namespace gmes::mt
{

mfem::Array<int> BuildEssentialBoundaryMask(const mfem::ParMesh& mesh)
{
    mfem::Array<int> ess_bdr;
    if (mesh.bdr_attributes.Size() == 0)
    {
        return ess_bdr;
    }
    ess_bdr.SetSize(mesh.bdr_attributes.Max());
    ess_bdr = 1;
    return ess_bdr;
}

MTBoundaryFieldCoefficient::MTBoundaryFieldCoefficient(
    Polarization polarization,
    double freq_hz,
    const MTInput& input,
    bool imaginary_part)
    : mfem::VectorCoefficient(3),
      polarization_(polarization),
      freq_hz_(freq_hz),
      input_(input),
      imaginary_part_(imaginary_part)
{
}

void MTBoundaryFieldCoefficient::Eval(mfem::Vector& value,
                                      mfem::ElementTransformation& transform,
                                      const mfem::IntegrationPoint& ip)
{
    value.SetSize(3);
    value = 0.0;

    mfem::Vector x(3);
    transform.Transform(ip, x);

    const double rho_bg = std::max(GetBackgroundRho(input_), 1.0e-12);
    const double omega = 2.0 * kPi * freq_hz_;
    const double mu = input_.mu0 * input_.solver.mu_r;
    const double skin_depth = std::sqrt(2.0 * rho_bg / std::max(omega * mu, 1.0e-30));
    const double depth = std::max(0.0, x[2]);
    const double attenuation = std::exp(-depth / skin_depth);
    const double phase = depth / skin_depth;
    const double scalar = attenuation *
        (imaginary_part_ ? -std::sin(phase) : std::cos(phase));

    if (polarization_ == Polarization::Ex)
    {
        value[0] = scalar;
    }
    else
    {
        value[1] = scalar;
    }
}

} // namespace gmes::mt
