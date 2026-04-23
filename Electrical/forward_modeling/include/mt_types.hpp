#pragma once

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace gmes::mt
{

constexpr double kPi = 3.141592653589793238462643383279502884;

enum class Polarization
{
    Ex = 0,
    Ey = 1
};

struct TensorModel
{
    int nx = 0;
    int ny = 0;
    int nz = 0;
    double dx = 1.0;
    double dy = 1.0;
    double dz = 1.0;
    std::vector<double> rho_ohm_m; // C-order: ((ix * ny) + iy) * nz + iz
};

struct PaddingConfig
{
    int npad_xy = 10;
    int npad_z = 10;
    double alpha = 1.4;
    bool enabled = true;
};

struct SolverConfig
{
    int order = 1;
    bool use_partial_assembly = false;
    double rel_tol = 1e-6;
    int max_iter = 2000;
    int krylov_dim = 200;
    bool verbose = true;
    std::string device = "cpu";
    double mu_r = 1.0;
};

struct ReceiverGrid
{
    std::vector<double> x;
    std::vector<double> y;
    double z = 0.0;
};

struct MTInput
{
    TensorModel model;
    PaddingConfig padding;
    SolverConfig solver;
    ReceiverGrid receivers;
    std::vector<double> freqs_hz;
    double mu0 = 4.0e-7 * kPi;
};

struct MTResponse
{
    int nfreq = 0;
    int nx = 0;
    int ny = 0;
    std::vector<double> app_res;   // shape: [nfreq, nx, ny, 2]
    std::vector<double> phase_deg; // shape: [nfreq, nx, ny, 2]
};

inline std::size_t TensorIndex(int ix, int iy, int iz, int ny, int nz)
{
    return (static_cast<std::size_t>(ix) * static_cast<std::size_t>(ny) +
            static_cast<std::size_t>(iy)) *
               static_cast<std::size_t>(nz) +
           static_cast<std::size_t>(iz);
}

inline std::size_t ResponseIndex(int ifreq, int ix, int iy, int component,
                                 int nx, int ny)
{
    return (((static_cast<std::size_t>(ifreq) * static_cast<std::size_t>(nx) +
              static_cast<std::size_t>(ix)) *
                 static_cast<std::size_t>(ny) +
             static_cast<std::size_t>(iy)) *
                2U) +
           static_cast<std::size_t>(component);
}

} // namespace gmes::mt
