#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

LOCAL_PREFIX="${GMES_MT_LOCAL_PREFIX:-${ROOT_DIR}/.local}"
HYPRE_PREFIX="${HYPRE_PREFIX:-${LOCAL_PREFIX}/hypre-cuda}"
CEED_PREFIX="${CEED_PREFIX:-${LOCAL_PREFIX}/libCEED-cuda}"
MFEM_PREFIX="${MFEM_PREFIX:-${LOCAL_PREFIX}/mfem-cuda}"

HYPRE_SRC="${HYPRE_SRC:-/tmp/hypre-cuda-src}"
CEED_SRC="${CEED_SRC:-/tmp/libCEED-cuda-src}"
MFEM_SRC="${MFEM_SRC:-/tmp/mfem-cuda-src}"

HYPRE_BUILD="${HYPRE_BUILD:-/tmp/hypre-cuda-build-gmes}"
MFEM_BUILD="${MFEM_BUILD:-/tmp/mfem-cuda-build-gmes}"
GMES_BUILD="${GMES_BUILD:-/tmp/gmes_forward_modeling_build_cuda}"

CUDA_DIR="${CUDA_DIR:-/usr/local/cuda-12.8}"
CUDA_ARCH_CMAKE="${CUDA_ARCH_CMAKE:-120}"
CUDA_ARCH_LIBCEED="${CUDA_ARCH_LIBCEED:-sm_120}"

HOST_CC="${HOST_CC:-/usr/bin/gcc-11}"
HOST_CXX="${HOST_CXX:-/usr/bin/g++-11}"
BUILD_JOBS="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Please activate the torch conda environment first."
  exit 1
fi

if [[ ! -x "${CONDA_PREFIX}/bin/mpicc" || ! -x "${CONDA_PREFIX}/bin/mpicxx" ]]; then
  echo "Expected MPI wrappers under ${CONDA_PREFIX}/bin."
  exit 1
fi

for path in "${HYPRE_SRC}" "${CEED_SRC}" "${MFEM_SRC}" "${CUDA_DIR}" "${HOST_CC}" "${HOST_CXX}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Required path not found: ${path}"
    exit 1
  fi
done

mkdir -p "${HYPRE_PREFIX}" "${CEED_PREFIX}" "${MFEM_PREFIX}"

export OMPI_CC="${OMPI_CC:-${HOST_CC}}"
export OMPI_CXX="${OMPI_CXX:-${HOST_CXX}}"

apply_mfem_low_memory_cuda_patches() {
  echo "Applying local low-memory MFEM CUDA patches"

  perl -0pi -e 's#lor/lor_batched\.cpp#lor/lor_batched_stub.cpp#g' \
    "${MFEM_SRC}/fem/CMakeLists.txt"

  cat > "${MFEM_SRC}/fem/lor/lor_batched_stub.cpp" <<'STUB'
// Low-memory CUDA build stub for batched LOR assembly.
//
// The full lor_batched.cpp translation unit instantiates a large set of CUDA
// kernels and can exceed the local nvcc memory limit. The MT H(curl) forward
// path does not use batched LOR assembly, so keep the public symbols available
// while reporting the optional optimization as unsupported.

#include "lor_batched.hpp"

namespace mfem
{

bool BatchedLORAssembly::FormIsSupported(BilinearForm &) { return false; }

void BatchedLORAssembly::FormLORVertexCoordinates(FiniteElementSpace &,
                                                  Vector &X_vert)
{
   X_vert.SetSize(0);
}

int BatchedLORAssembly::FillI(SparseMatrix &) const
{
   MFEM_ABORT("Batched LOR assembly is disabled in this low-memory CUDA build.");
   return 0;
}

void BatchedLORAssembly::FillJAndData(SparseMatrix &)
{
   MFEM_ABORT("Batched LOR assembly is disabled in this low-memory CUDA build.");
}

void BatchedLORAssembly::SparseIJToCSR_DG(OperatorHandle &) const
{
   MFEM_ABORT("Batched LOR assembly is disabled in this low-memory CUDA build.");
}

void BatchedLORAssembly::SparseIJToCSR(OperatorHandle &) const
{
   MFEM_ABORT("Batched LOR assembly is disabled in this low-memory CUDA build.");
}

void BatchedLORAssembly::AssembleWithoutBC(BilinearForm &, OperatorHandle &)
{
   MFEM_ABORT("Batched LOR assembly is disabled in this low-memory CUDA build.");
}

#ifdef MFEM_USE_MPI
void BatchedLORAssembly::ParAssemble_DG(SparseMatrix &, OperatorHandle &)
{
   MFEM_ABORT("Batched LOR assembly is disabled in this low-memory CUDA build.");
}

void BatchedLORAssembly::ParAssemble(BilinearForm &, const Array<int> &,
                                     OperatorHandle &)
{
   MFEM_ABORT("Batched LOR assembly is disabled in this low-memory CUDA build.");
}
#endif

void BatchedLORAssembly::Assemble(BilinearForm &, const Array<int>,
                                  OperatorHandle &)
{
   MFEM_ABORT("Batched LOR assembly is disabled in this low-memory CUDA build.");
}

BatchedLORAssembly::BatchedLORAssembly(FiniteElementSpace &fes_ho_)
   : fes_ho(fes_ho_)
{ }

IntegrationRule GetLobattoIntRule(Geometry::Type geom, int nd1d)
{
   return IntRules.Get(geom, 2*nd1d - 3);
}

IntegrationRule GetCollocatedIntRule(FiniteElementSpace &fes)
{
   const Geometry::Type geom = fes.GetMesh()->GetTypicalElementGeometry();
   return GetLobattoIntRule(geom, fes.GetMaxElementOrder() + 1);
}

IntegrationRule GetCollocatedFaceIntRule(FiniteElementSpace &fes)
{
   const Geometry::Type geom = fes.GetMesh()->GetTypicalFaceGeometry();
   return GetLobattoIntRule(geom, fes.GetMaxElementOrder() + 1);
}

}
STUB

  perl -0pi -e 's#switch \(\(dofs1D << 4 \) \| quad1D\)\s*\{.*?default:\s*return EADiffusionAssemble1D\(ne,B,G,pa_data,ea_data,add,\s*dofs1D,quad1D\);\s*\}#return EADiffusionAssemble1D(ne,B,G,pa_data,ea_data,add,\n                                   dofs1D,quad1D);#s' \
    "${MFEM_SRC}/fem/integ/bilininteg_diffusion_ea.cpp"
  perl -0pi -e 's#switch \(\(dofs1D << 4 \) \| quad1D\)\s*\{.*?default:\s*return EADiffusionAssemble2D\(ne,B,G,pa_data,ea_data,add,\s*dofs1D,quad1D\);\s*\}#return EADiffusionAssemble2D(ne,B,G,pa_data,ea_data,add,\n                                   dofs1D,quad1D);#s' \
    "${MFEM_SRC}/fem/integ/bilininteg_diffusion_ea.cpp"
  perl -0pi -e 's#switch \(\(dofs1D << 4 \) \| quad1D\)\s*\{.*?default:\s*return EADiffusionAssemble3D\(ne,B,G,pa_data,ea_data,add,\s*dofs1D,quad1D\);\s*\}#return EADiffusionAssemble3D(ne,B,G,pa_data,ea_data,add,\n                                   dofs1D,quad1D);#s' \
    "${MFEM_SRC}/fem/integ/bilininteg_diffusion_ea.cpp"

  perl -0pi -e 's#const int id = \(D1D << 4\) \| Q1D;\s*##' \
    "${MFEM_SRC}/fem/integ/bilininteg_hdiv_kernels.cpp"
  perl -0pi -e 's#if \(dim == 2\)\s*\{\s*switch \(id\)\s*\{.*?default: // fallback\s*return PAHdivMassApply2D\(D1D,Q1D,NE,symmetric,Bo,Bc,Bot,Bct,op,x,y\);\s*\}\s*\}#if (dim == 2)\n   {\n      return PAHdivMassApply2D(D1D,Q1D,NE,symmetric,Bo,Bc,Bot,Bct,op,x,y);\n   }#s' \
    "${MFEM_SRC}/fem/integ/bilininteg_hdiv_kernels.cpp"
  perl -0pi -e 's#else if \(dim == 3\)\s*\{\s*switch \(id\)\s*\{.*?default: // fallback\s*return PAHdivMassApply3D\(D1D,Q1D,NE,symmetric,Bo,Bc,Bot,Bct,op,x,y\);\s*\}\s*\}#else if (dim == 3)\n   {\n      return PAHdivMassApply3D(D1D,Q1D,NE,symmetric,Bo,Bc,Bot,Bct,op,x,y);\n   }#s' \
    "${MFEM_SRC}/fem/integ/bilininteg_hdiv_kernels.cpp"

  perl -0pi -e 's#void InitTensorEvalHDivKernels\(\)\s*\{.*?\n\}#void InitTensorEvalHDivKernels()\n{\n   // Disabled in the local low-memory CUDA build. H(div) quadrature\n   // interpolation remains available through TensorEvalHDivKernels::Fallback.\n}#s' \
    "${MFEM_SRC}/fem/qinterp/eval_hdiv.cpp"
}

echo "[1/4] Configure and build hypre with CUDA"
cmake -S "${HYPRE_SRC}/src" -B "${HYPRE_BUILD}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DHYPRE_ENABLE_MPI=ON \
  -DHYPRE_ENABLE_CUDA=ON \
  -DHYPRE_ENABLE_UMPIRE=OFF \
  -DHYPRE_BUILD_UMPIRE=OFF \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH_CMAKE}" \
  -DCMAKE_INSTALL_PREFIX="${HYPRE_PREFIX}" \
  -DCMAKE_C_COMPILER="${HOST_CC}" \
  -DCMAKE_CXX_COMPILER="${HOST_CXX}" \
  -DCMAKE_CUDA_HOST_COMPILER="${HOST_CXX}" \
  -DMPI_C_COMPILER="${CONDA_PREFIX}/bin/mpicc" \
  -DMPI_CXX_COMPILER="${CONDA_PREFIX}/bin/mpicxx"
cmake --build "${HYPRE_BUILD}" -j "${BUILD_JOBS}"
cmake --install "${HYPRE_BUILD}"

echo "[2/4] Build and install libCEED with CUDA backends"
make -C "${CEED_SRC}" \
  CC="${HOST_CC}" \
  CXX="${HOST_CXX}" \
  CUDA_DIR="${CUDA_DIR}" \
  CUDA_ARCH="${CUDA_ARCH_LIBCEED}" \
  prefix="${CEED_PREFIX}" \
  -j "${BUILD_JOBS}"
make -C "${CEED_SRC}" \
  CC="${HOST_CC}" \
  CXX="${HOST_CXX}" \
  CUDA_DIR="${CUDA_DIR}" \
  CUDA_ARCH="${CUDA_ARCH_LIBCEED}" \
  prefix="${CEED_PREFIX}" \
  install

echo "[3/4] Configure, build, and install MFEM with CUDA + CEED"
apply_mfem_low_memory_cuda_patches
MFEM_PREFIX_PATH="${HYPRE_PREFIX};${CEED_PREFIX};${CONDA_PREFIX}"
cmake -S "${MFEM_SRC}" -B "${MFEM_BUILD}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_INSTALL_PREFIX="${MFEM_PREFIX}" \
  -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
  -DMFEM_USE_MPI=YES \
  -DMFEM_USE_METIS=OFF \
  -DMFEM_USE_CUDA=YES \
  -DMFEM_USE_CEED=YES \
  -DMFEM_ENABLE_TESTING=OFF \
  -DMFEM_ENABLE_EXAMPLES=OFF \
  -DMFEM_ENABLE_MINIAPPS=OFF \
  -DMFEM_ENABLE_BENCHMARKS=OFF \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH_CMAKE}" \
  -DCMAKE_CUDA_HOST_COMPILER="${HOST_CXX}" \
  '-DCMAKE_CUDA_FLAGS_RELEASE=-O1 -DNDEBUG' \
  "-DCMAKE_CUDA_FLAGS=-I${CONDA_PREFIX}/include" \
  -DMPICXX="${CONDA_PREFIX}/bin/mpicxx" \
  -DMPI_CXX_COMPILER="${CONDA_PREFIX}/bin/mpicxx" \
  -DMPI_CXX_INCLUDE_DIRS="${CONDA_PREFIX}/include" \
  -DMPI_CXX_COMPILER_INCLUDE_DIRS="${CONDA_PREFIX}/include" \
  -DHYPRE_DIR="${HYPRE_PREFIX}" \
  -DCEED_DIR="${CEED_PREFIX}" \
  "-DCMAKE_PREFIX_PATH=${MFEM_PREFIX_PATH}"
cmake --build "${MFEM_BUILD}" -j "${BUILD_JOBS}"
cmake --install "${MFEM_BUILD}"

echo "[4/4] Rebuild mt_forward_mfem against the local CUDA-enabled MFEM"
GMES_PREFIX_PATH="${MFEM_PREFIX};${CONDA_PREFIX}"
GMES_RPATH="${MFEM_PREFIX}/lib;${HYPRE_PREFIX}/lib;${CEED_PREFIX}/lib;${CONDA_PREFIX}/lib;${CUDA_DIR}/lib64"
cmake -S "${ROOT_DIR}" -B "${GMES_BUILD}" \
  "-DCMAKE_PREFIX_PATH=${GMES_PREFIX_PATH}" \
  -DMFEM_DIR="${MFEM_PREFIX}/lib/cmake/mfem" \
  -DCMAKE_BUILD_RPATH="${GMES_RPATH}" \
  -DPython3_EXECUTABLE="${CONDA_PREFIX}/bin/python" \
  -Dpybind11_DIR="${CONDA_PREFIX}/share/cmake/pybind11" \
  -DMPI_CXX_COMPILER="${CONDA_PREFIX}/bin/mpicxx" \
  -DCMAKE_CXX_COMPILER="${CONDA_PREFIX}/bin/mpicxx"
cmake --build "${GMES_BUILD}" -j "${BUILD_JOBS}"
ctest --test-dir "${GMES_BUILD}" --output-on-failure

echo
echo "CUDA-enabled stack and mt_forward_mfem build complete."
echo "MFEM prefix: ${MFEM_PREFIX}"
echo "Backend module directory: ${ROOT_DIR}"
