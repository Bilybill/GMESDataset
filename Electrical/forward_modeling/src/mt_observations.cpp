#include "mt_observations.hpp"

#include "mt_background_1d.hpp"
#include "mt_config.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

namespace gmes::mt
{

namespace
{

std::complex<double> MagneticFromCurl(std::complex<double> curl_component,
                                      double omega_mu)
{
    return std::complex<double>(-curl_component.imag(), curl_component.real()) /
           omega_mu;
}

double PhaseDeg(std::complex<double> z)
{
    return std::atan2(z.imag(), z.real()) * 180.0 / kPi;
}

} // namespace

void AccumulateSurfaceResponse(mfem::ParMesh& mesh,
                               const mfem::ParFiniteElementSpace&,
                               const mfem::ParComplexGridFunction& field,
                               double freq_hz,
                               Polarization polarization,
                               const MTInput& input,
                               const TensorMeshAxes& axes,
                               MTResponse& output,
                               int freq_index)
{
    const int component = polarization == Polarization::Ex ? 0 : 1;
    const double rho_bg = GetBackgroundRho(input);
    const int npoints = output.nx * output.ny;

    mfem::DenseMatrix point_mat(3, npoints);
    const int npad_xy = input.padding.enabled ? input.padding.npad_xy : 0;
    const double x0 = axes.x_edges[static_cast<std::size_t>(npad_xy)];
    const double y0 = axes.y_edges[static_cast<std::size_t>(npad_xy)];
    const double z_eps = std::max(1.0e-9, input.model.dz * 1.0e-6);

    for (int ix = 0, p = 0; ix < output.nx; ++ix)
    {
        for (int iy = 0; iy < output.ny; ++iy, ++p)
        {
            const double rx = ix < static_cast<int>(input.receivers.x.size())
                ? input.receivers.x[static_cast<std::size_t>(ix)]
                : (static_cast<double>(ix) + 0.5) * input.model.dx;
            const double ry = iy < static_cast<int>(input.receivers.y.size())
                ? input.receivers.y[static_cast<std::size_t>(iy)]
                : (static_cast<double>(iy) + 0.5) * input.model.dy;

            point_mat(0, p) = x0 + rx;
            point_mat(1, p) = y0 + ry;
            point_mat(2, p) = axes.z_edges.front() + input.receivers.z + z_eps;
        }
    }

    mfem::Array<int> elem_ids;
    mfem::Array<mfem::IntegrationPoint> ips;
    mesh.FindPoints(point_mat, elem_ids, ips, false);

    mfem::CurlGridFunctionCoefficient curl_re(&field.real());
    mfem::CurlGridFunctionCoefficient curl_im(&field.imag());

    std::vector<double> local_app(static_cast<std::size_t>(npoints), 0.0);
    std::vector<double> local_phase(static_cast<std::size_t>(npoints), 0.0);
    std::vector<int> local_count(static_cast<std::size_t>(npoints), 0);

    const double omega = 2.0 * kPi * freq_hz;
    const double omega_mu = omega * input.mu0 * input.solver.mu_r;

    mfem::Vector e_re(3), e_im(3), curl_real(3), curl_imag(3);
    for (int p = 0; p < npoints; ++p)
    {
        const int elem = elem_ids[p];
        if (elem < 0)
        {
            continue;
        }

        field.real().GetVectorValue(elem, ips[p], e_re);
        field.imag().GetVectorValue(elem, ips[p], e_im);

        mfem::ElementTransformation* transform =
            mesh.GetElementTransformation(elem);
        curl_re.Eval(curl_real, *transform, ips[p]);
        curl_im.Eval(curl_imag, *transform, ips[p]);

        const int e_comp = polarization == Polarization::Ex ? 0 : 1;
        const int h_comp = polarization == Polarization::Ex ? 1 : 0;
        const std::complex<double> electric(e_re[e_comp], e_im[e_comp]);
        const std::complex<double> curl_component(curl_real[h_comp],
                                                  curl_imag[h_comp]);
        const std::complex<double> magnetic =
            MagneticFromCurl(curl_component, omega_mu);

        std::complex<double> impedance(0.0, 0.0);
        if (std::abs(magnetic) > 1.0e-30)
        {
            impedance = electric / magnetic;
        }

        local_app[static_cast<std::size_t>(p)] =
            std::norm(impedance) / std::max(omega * input.mu0, 1.0e-30);
        local_phase[static_cast<std::size_t>(p)] = PhaseDeg(impedance);
        local_count[static_cast<std::size_t>(p)] = 1;
    }

    std::vector<double> global_app(static_cast<std::size_t>(npoints), 0.0);
    std::vector<double> global_phase(static_cast<std::size_t>(npoints), 0.0);
    std::vector<int> global_count(static_cast<std::size_t>(npoints), 0);

    MPI_Allreduce(local_app.data(), global_app.data(), npoints, MPI_DOUBLE,
                  MPI_SUM, mesh.GetComm());
    MPI_Allreduce(local_phase.data(), global_phase.data(), npoints, MPI_DOUBLE,
                  MPI_SUM, mesh.GetComm());
    MPI_Allreduce(local_count.data(), global_count.data(), npoints, MPI_INT,
                  MPI_SUM, mesh.GetComm());

    for (int ix = 0, p = 0; ix < output.nx; ++ix)
    {
        for (int iy = 0; iy < output.ny; ++iy, ++p)
        {
            const std::size_t out_idx =
                ResponseIndex(freq_index, ix, iy, component, output.nx, output.ny);
            if (global_count[static_cast<std::size_t>(p)] > 0)
            {
                output.app_res[out_idx] = global_app[static_cast<std::size_t>(p)];
                output.phase_deg[out_idx] =
                    global_phase[static_cast<std::size_t>(p)];
            }
            else
            {
                output.app_res[out_idx] = rho_bg;
                output.phase_deg[out_idx] = polarization == Polarization::Ex
                    ? kDefaultHalfspacePhaseDeg
                    : -kDefaultHalfspacePhaseDeg;
            }
        }
    }
}

} // namespace gmes::mt
