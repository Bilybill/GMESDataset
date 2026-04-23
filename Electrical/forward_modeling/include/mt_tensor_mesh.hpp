#pragma once

#include "mt_types.hpp"
#include "mfem.hpp"

#include <memory>

namespace gmes::mt
{

struct TensorMeshAxes
{
    std::vector<double> x_edges;
    std::vector<double> y_edges;
    std::vector<double> z_edges;
};

TensorMeshAxes BuildTensorMeshAxes(const MTInput& input);

std::unique_ptr<mfem::Mesh> BuildTensorHexMesh(const MTInput& input,
                                               TensorMeshAxes* axes);

} // namespace gmes::mt
