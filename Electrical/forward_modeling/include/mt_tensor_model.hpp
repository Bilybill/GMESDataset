#pragma once

#include "mt_types.hpp"
#include "mfem.hpp"

namespace gmes::mt
{

class TensorSigmaCoefficient : public mfem::Coefficient
{
public:
    TensorSigmaCoefficient(const MTInput& input,
                           const std::vector<double>& x_edges,
                           const std::vector<double>& y_edges,
                           const std::vector<double>& z_edges);

    double Eval(mfem::ElementTransformation& transform,
                const mfem::IntegrationPoint& ip) override;

private:
    const MTInput& input_;
    const std::vector<double>& x_edges_;
    const std::vector<double>& y_edges_;
    const std::vector<double>& z_edges_;
    double background_rho_ = 1.0;

    double RhoAtCell(int ix, int iy, int iz) const;
};

class FrequencyScaledSigmaCoefficient : public mfem::Coefficient
{
public:
    FrequencyScaledSigmaCoefficient(double scale, mfem::Coefficient& sigma);

    double Eval(mfem::ElementTransformation& transform,
                const mfem::IntegrationPoint& ip) override;

private:
    double scale_ = 1.0;
    mfem::Coefficient& sigma_;
};

} // namespace gmes::mt
