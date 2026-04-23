#include "mt_tensor_mesh.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gmes::mt
{

namespace
{

std::vector<double> BuildAxis(int n_core, double h, int npad_before,
                              int npad_after, double alpha)
{
    std::vector<double> widths;
    widths.reserve(static_cast<std::size_t>(n_core + npad_before + npad_after));

    for (int i = npad_before - 1; i >= 0; --i)
    {
        widths.push_back(h * std::pow(alpha, static_cast<double>(i + 1)));
    }
    for (int i = 0; i < n_core; ++i)
    {
        widths.push_back(h);
    }
    for (int i = 0; i < npad_after; ++i)
    {
        widths.push_back(h * std::pow(alpha, static_cast<double>(i + 1)));
    }

    std::vector<double> edges(widths.size() + 1U, 0.0);
    for (std::size_t i = 0; i < widths.size(); ++i)
    {
        edges[i + 1U] = edges[i] + widths[i];
    }
    return edges;
}

int CoordinateToIndex(double value, double old_max, int n)
{
    if (old_max <= 0.0)
    {
        return 0;
    }
    const double scaled = value / old_max * static_cast<double>(n);
    return std::clamp(static_cast<int>(std::llround(scaled)), 0, n);
}

} // namespace

TensorMeshAxes BuildTensorMeshAxes(const MTInput& input)
{
    const int npad_xy = input.padding.enabled ? input.padding.npad_xy : 0;
    const int npad_z = input.padding.enabled ? input.padding.npad_z : 0;
    const double alpha = input.padding.enabled ? input.padding.alpha : 1.0;

    TensorMeshAxes axes;
    axes.x_edges = BuildAxis(input.model.nx, input.model.dx, npad_xy, npad_xy, alpha);
    axes.y_edges = BuildAxis(input.model.ny, input.model.dy, npad_xy, npad_xy, alpha);
    axes.z_edges = BuildAxis(input.model.nz, input.model.dz, 0, npad_z, alpha);
    return axes;
}

std::unique_ptr<mfem::Mesh> BuildTensorHexMesh(const MTInput& input,
                                               TensorMeshAxes* axes_out)
{
    TensorMeshAxes axes = BuildTensorMeshAxes(input);

    const int nx = static_cast<int>(axes.x_edges.size()) - 1;
    const int ny = static_cast<int>(axes.y_edges.size()) - 1;
    const int nz = static_cast<int>(axes.z_edges.size()) - 1;
    if (nx <= 0 || ny <= 0 || nz <= 0)
    {
        throw std::runtime_error("Invalid padded tensor mesh dimensions.");
    }

    auto mesh = std::make_unique<mfem::Mesh>(
        mfem::Mesh::MakeCartesian3D(nx, ny, nz, mfem::Element::HEXAHEDRON,
                                    1.0, 1.0, 1.0, true));

    for (int iv = 0; iv < mesh->GetNV(); ++iv)
    {
        double* vertex = mesh->GetVertex(iv);
        const int ix = CoordinateToIndex(vertex[0], 1.0, nx);
        const int iy = CoordinateToIndex(vertex[1], 1.0, ny);
        const int iz = CoordinateToIndex(vertex[2], 1.0, nz);
        vertex[0] = axes.x_edges[static_cast<std::size_t>(ix)];
        vertex[1] = axes.y_edges[static_cast<std::size_t>(iy)];
        vertex[2] = axes.z_edges[static_cast<std::size_t>(iz)];
    }

    mesh->SetCurvature(1, false, 3, mfem::Ordering::byNODES);

    if (axes_out != nullptr)
    {
        *axes_out = std::move(axes);
    }
    return mesh;
}

} // namespace gmes::mt
