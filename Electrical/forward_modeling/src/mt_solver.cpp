#include "mt_solver.hpp"

#include "mt_boundary_conditions.hpp"
#include "mt_driver.hpp"
#include "mt_observations.hpp"
#include "mt_tensor_mesh.hpp"
#include "mt_tensor_model.hpp"

#include "mfem.hpp"
#include <mpi.h>

#include <cctype>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace gmes::mt
{

using namespace mfem;

namespace
{

void EnsureMPIInitialized()
{
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized)
    {
        int provided = 0;
        int argc = 0;
        char** argv = nullptr;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    }
}

bool DeviceStringRequestsCuda(std::string device)
{
    for (char& ch : device)
    {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return device.find("cuda") != std::string::npos;
}

} // namespace

struct MT3DForwardSolver::Impl
{
    explicit Impl(MTInput in)
        : input(NormalizeInput(std::move(in)))
    {
        EnsureMPIInitialized();
#ifndef MFEM_USE_CUDA
        if (DeviceStringRequestsCuda(input.solver.device))
        {
            throw std::runtime_error(
                "MFEM device 'cuda' was requested, but the linked MFEM library "
                "was built without MFEM_USE_CUDA=YES. Rebuild MFEM/hypre with "
                "CUDA support, then rebuild mt_forward_mfem.");
        }
#endif
        device.Configure(input.solver.device.c_str());
        if (input.solver.verbose)
        {
            device.Print();
        }

        serial_mesh = BuildTensorHexMesh(input, &axes);
        pmesh = std::make_unique<ParMesh>(MPI_COMM_WORLD, *serial_mesh);

        fec = std::make_unique<ND_FECollection>(input.solver.order, 3);
        fes = std::make_unique<ParFiniteElementSpace>(pmesh.get(), fec.get());
    }

    MTInput input;
    Device device;
    TensorMeshAxes axes;

    std::unique_ptr<Mesh> serial_mesh;
    std::unique_ptr<ParMesh> pmesh;
    std::unique_ptr<ND_FECollection> fec;
    std::unique_ptr<ParFiniteElementSpace> fes;
};

MT3DForwardSolver::MT3DForwardSolver(const MTInput& input)
    : impl_(std::make_unique<Impl>(input))
{
}

MT3DForwardSolver::~MT3DForwardSolver() = default;

MTResponse MT3DForwardSolver::Run()
{
    if (impl_->input.solver.use_partial_assembly)
    {
        throw std::runtime_error(
            "Partial assembly is intentionally disabled in the first MFEM migration slice. "
            "Use assembled operators with HypreAMS for low-frequency robustness.");
    }

    Array<int> ess_bdr = BuildEssentialBoundaryMask(*impl_->pmesh);
    Array<int> ess_tdof_list;
    impl_->fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

    MTResponse out;
    out.nfreq = static_cast<int>(impl_->input.freqs_hz.size());
    out.nx = impl_->input.model.nx;
    out.ny = impl_->input.model.ny;
    out.app_res.assign(static_cast<std::size_t>(out.nfreq) *
                           static_cast<std::size_t>(out.nx) *
                           static_cast<std::size_t>(out.ny) * 2U,
                       0.0);
    out.phase_deg.assign(out.app_res.size(), 0.0);

    const ComplexOperator::Convention convention =
        ComplexOperator::BLOCK_SYMMETRIC;

    ConstantCoefficient curl_coeff(1.0 / impl_->input.solver.mu_r);
    TensorSigmaCoefficient sigma_coeff(impl_->input,
                                       impl_->axes.x_edges,
                                       impl_->axes.y_edges,
                                       impl_->axes.z_edges);

    for (int ifreq = 0; ifreq < out.nfreq; ++ifreq)
    {
        const double freq_hz = impl_->input.freqs_hz[ifreq];
        const double omega = 2.0 * kPi * freq_hz;

        FrequencyScaledSigmaCoefficient omega_mu_sigma(
            omega * impl_->input.mu0 * impl_->input.solver.mu_r,
            sigma_coeff);

        for (const Polarization pol : {Polarization::Ex, Polarization::Ey})
        {
            ParComplexLinearForm rhs(impl_->fes.get(), convention);
            rhs = 0.0;
            rhs.Assemble();

            ParComplexGridFunction field(impl_->fes.get());
            field = 0.0;

            MTBoundaryFieldCoefficient bc_re(pol, freq_hz, impl_->input, false);
            MTBoundaryFieldCoefficient bc_im(pol, freq_hz, impl_->input, true);
            field.ProjectBdrCoefficientTangent(bc_re, bc_im, ess_bdr);

            ParSesquilinearForm form(impl_->fes.get(), convention);
            form.AddDomainIntegrator(new CurlCurlIntegrator(curl_coeff), nullptr);
            form.AddDomainIntegrator(nullptr,
                                     new VectorFEMassIntegrator(omega_mu_sigma));
            form.Assemble();

            OperatorPtr system_operator;
            Vector B, X;
            form.FormLinearSystem(ess_tdof_list, field, rhs, system_operator, X, B);

            ParBilinearForm preconditioner_form(impl_->fes.get());
            preconditioner_form.AddDomainIntegrator(
                new CurlCurlIntegrator(curl_coeff));
            preconditioner_form.AddDomainIntegrator(
                new VectorFEMassIntegrator(omega_mu_sigma));
            preconditioner_form.Assemble();
            preconditioner_form.Finalize();

            OperatorPtr preconditioner_operator;
            preconditioner_form.FormSystemMatrix(ess_tdof_list,
                                                 preconditioner_operator);

            HypreParMatrix* assembled_matrix =
                preconditioner_operator.As<HypreParMatrix>();
            if (assembled_matrix == nullptr)
            {
                throw std::runtime_error(
                    "Expected an assembled HypreParMatrix for HypreAMS.");
            }

            HypreAMS ams(*assembled_matrix, impl_->fes.get());

            Array<int> block_offsets(3);
            block_offsets[0] = 0;
            block_offsets[1] = impl_->fes->GetTrueVSize();
            block_offsets[2] = impl_->fes->GetTrueVSize();
            block_offsets.PartialSum();

            BlockDiagonalPreconditioner block_preconditioner(block_offsets);
            block_preconditioner.SetDiagonalBlock(0, &ams);
            block_preconditioner.SetDiagonalBlock(1, &ams);

            GMRESSolver gmres(MPI_COMM_WORLD);
            gmres.SetPrintLevel(impl_->input.solver.verbose ? 1 : 0);
            gmres.SetKDim(impl_->input.solver.krylov_dim);
            gmres.SetMaxIter(impl_->input.solver.max_iter);
            gmres.SetRelTol(impl_->input.solver.rel_tol);
            gmres.SetAbsTol(0.0);
            gmres.SetOperator(*system_operator);
            gmres.SetPreconditioner(block_preconditioner);
            gmres.Mult(B, X);

            if (!gmres.GetConverged())
            {
                throw std::runtime_error(
                    "MFEM GMRES failed to converge for one MT polarization.");
            }

            form.RecoverFEMSolution(X, rhs, field);

            AccumulateSurfaceResponse(*impl_->pmesh,
                                      *impl_->fes,
                                      field,
                                      freq_hz,
                                      pol,
                                      impl_->input,
                                      impl_->axes,
                                      out,
                                      ifreq);
        }
    }

    return out;
}

MTResponse ComputeMT3D(const MTInput& input)
{
    MT3DForwardSolver solver(input);
    return solver.Run();
}

} // namespace gmes::mt
