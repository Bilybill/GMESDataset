#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mt_driver.hpp"
#include "mt_solver.hpp"

#include "mfem.hpp"

#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace py = pybind11;
using namespace gmes::mt;

namespace
{

MTInput MakeInput(py::array_t<double, py::array::c_style | py::array::forcecast> rho,
                  double dx,
                  double dy,
                  double dz,
                  std::vector<double> freqs_hz,
                  int npad_xy,
                  int npad_z,
                  double alpha,
                  double rel_tol,
                  int max_iter,
                  bool verbose,
                  const std::string& device)
{
    py::buffer_info buf = rho.request();
    if (buf.ndim != 3)
    {
        throw std::runtime_error("rho must be a 3D array with shape (NX, NY, NZ).");
    }

    MTInput input;
    input.model.nx = static_cast<int>(buf.shape[0]);
    input.model.ny = static_cast<int>(buf.shape[1]);
    input.model.nz = static_cast<int>(buf.shape[2]);
    input.model.dx = dx;
    input.model.dy = dy;
    input.model.dz = dz;

    const std::size_t n =
        static_cast<std::size_t>(input.model.nx) *
        static_cast<std::size_t>(input.model.ny) *
        static_cast<std::size_t>(input.model.nz);

    const auto* data = static_cast<const double*>(buf.ptr);
    input.model.rho_ohm_m.assign(data, data + n);
    input.freqs_hz = std::move(freqs_hz);

    input.padding.npad_xy = npad_xy;
    input.padding.npad_z = npad_z;
    input.padding.alpha = alpha;
    input.padding.enabled = npad_xy > 0 || npad_z > 0;

    input.solver.rel_tol = rel_tol;
    input.solver.max_iter = max_iter;
    input.solver.verbose = verbose;
    input.solver.device = device;
    input = NormalizeInput(std::move(input));
    return input;
}

} // namespace

PYBIND11_MODULE(mt_forward_mfem, m)
{
    m.doc() = "MFEM backend for GMESDataset 3D MT forward modeling";

    m.def("is_cuda_enabled",
          []()
          {
#ifdef MFEM_USE_CUDA
              return true;
#else
              return false;
#endif
          },
          "Return true when the linked MFEM library was built with CUDA.");

    m.def("compute_mt_3d",
          [](py::array_t<double, py::array::c_style | py::array::forcecast> rho,
             double dx,
             double dy,
             double dz,
             std::vector<double> freqs_hz,
             int npad_xy,
             int npad_z,
             double alpha,
             double rel_tol,
             int max_iter,
             bool verbose,
             const std::string& device)
          {
              MTInput input = MakeInput(rho, dx, dy, dz, std::move(freqs_hz),
                                        npad_xy, npad_z, alpha, rel_tol,
                                        max_iter, verbose, device);
              MTResponse output = ComputeMT3D(input);

              py::array_t<double> app_res(
                  std::vector<py::ssize_t>{output.nfreq, output.nx, output.ny, 2});
              py::array_t<double> phase(
                  std::vector<py::ssize_t>{output.nfreq, output.nx, output.ny, 2});

              std::memcpy(app_res.mutable_data(),
                          output.app_res.data(),
                          output.app_res.size() * sizeof(double));
              std::memcpy(phase.mutable_data(),
                          output.phase_deg.data(),
                          output.phase_deg.size() * sizeof(double));

              return py::make_tuple(app_res, phase);
          },
          py::arg("rho"),
          py::arg("dx"),
          py::arg("dy"),
          py::arg("dz"),
          py::arg("freqs_hz"),
          py::arg("npad_xy") = 10,
          py::arg("npad_z") = 10,
          py::arg("alpha") = 1.4,
          py::arg("rel_tol") = 1e-6,
          py::arg("max_iter") = 2000,
          py::arg("verbose") = false,
          py::arg("device") = "cpu");
}
