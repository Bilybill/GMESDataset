#pragma once

#include "mt_types.hpp"
#include "mfem.hpp"

namespace gmes::mt
{

mfem::Array<int> BuildEssentialBoundaryMask(const mfem::ParMesh& mesh);

class MTBoundaryFieldCoefficient : public mfem::VectorCoefficient
{
public:
    MTBoundaryFieldCoefficient(Polarization polarization,
                               double freq_hz,
                               const MTInput& input,
                               bool imaginary_part);

    void Eval(mfem::Vector& value,
              mfem::ElementTransformation& transform,
              const mfem::IntegrationPoint& ip) override;

private:
    Polarization polarization_;
    double freq_hz_ = 1.0;
    const MTInput& input_;
    bool imaginary_part_ = false;
};

} // namespace gmes::mt
