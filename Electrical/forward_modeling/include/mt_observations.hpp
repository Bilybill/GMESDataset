#pragma once

#include "mt_tensor_mesh.hpp"
#include "mt_types.hpp"
#include "mfem.hpp"

namespace gmes::mt
{

void AccumulateSurfaceResponse(mfem::ParMesh& mesh,
                               const mfem::ParFiniteElementSpace& fes,
                               const mfem::ParComplexGridFunction& field,
                               double freq_hz,
                               Polarization polarization,
                               const MTInput& input,
                               const TensorMeshAxes& axes,
                               MTResponse& output,
                               int freq_index);

} // namespace gmes::mt
