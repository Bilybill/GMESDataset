# MFEM MT Forward Backend

This directory is being migrated from a custom CUDA MT backend to an MFEM-based
backend. The public Python facade remains `mt_forward.py`; callers should keep
using `MTForward3D`.

The new backend target is `mt_forward_mfem`. It is designed around MFEM's
parallel H(curl) stack:

- `ParMesh`
- `ND_FECollection`
- `ParFiniteElementSpace`
- `ParComplexGridFunction`
- `ParSesquilinearForm`
- `HypreAMS`
- `GMRESSolver`

The first implementation uses assembled operators and the scaled system

```text
Kcurl + i * omega * mu0 * sigma * M
```

instead of the legacy `sigma M + 1 / (i omega mu) Kcurl` scaling.

## Local Conda Build

MFEM and pybind11 are external dependencies and are not vendored here.
The current tested local setup uses the `laphg` conda environment:

```bash
conda install -n laphg -c conda-forge mfem=4.8=mpi_openmpi\* gxx_linux-64 cmake pybind11

export CONDA_PREFIX=/home/wangyh/anaconda3/envs/laphg
export PATH="$CONDA_PREFIX/bin:$PATH"

cmake -S GMESDataset/Electrical/forward_modeling -B /tmp/gmes_forward_modeling_build \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  -DCMAKE_CXX_COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++" \
  -DMPI_CXX_COMPILER="$CONDA_PREFIX/bin/mpicxx" \
  -DPython3_EXECUTABLE="$CONDA_PREFIX/bin/python" \
  -Dpybind11_DIR="$CONDA_PREFIX/share/cmake/pybind11"

cmake --build /tmp/gmes_forward_modeling_build -j
ctest --test-dir /tmp/gmes_forward_modeling_build --output-on-failure
$CONDA_PREFIX/bin/python -m pytest GMESDataset/Electrical/forward_modeling/tests -q
```

For the `torch` conda environment, use the same CMake command with
`CONDA_PREFIX=/home/wangyh/anaconda3/envs/torch` and build into a separate
directory such as `/tmp/gmes_forward_modeling_build_torch`. Pip-installed torch
can load the system `libstdc++.so.6` before MFEM; set the runtime library path
before launching Python if scripts import torch first:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

## MFEM Device Selection

The Python facade passes MFEM's device string through to the C++ backend:

```python
from mt_forward import MTForward3D, mfem_cuda_enabled, mfem_ceed_enabled

print(mfem_cuda_enabled())
print(mfem_ceed_enabled())
op = MTForward3D(freqs=[1.0], dx=100.0, dy=100.0, dz=100.0, device="cuda")
```

The same setting can be provided through the environment:

```bash
export GMES_MT_MFEM_DEVICE=cuda
```

The debug script also accepts:

```bash
python legacy_cuda/test_forward_debug.py --mfem-device cuda
```

This only selects a CUDA backend when the linked MFEM and hypre libraries were
built with CUDA. A CPU-only MFEM package will reject `device="cuda"` at runtime.

For a libCEED-backed device path use MFEM's CEED device strings together with
partial assembly:

```python
op = MTForward3D(
    freqs=[1.0],
    dx=100.0,
    dy=100.0,
    dz=100.0,
    device="ceed-cuda:/gpu/cuda/shared",
    use_partial_assembly=True,
)
```

The debug script exposes the same route:

```bash
python legacy_cuda/test_forward_debug.py \
  --mfem-device "ceed-cuda:/gpu/cuda/shared" \
  --partial-assembly
```

`device="ceed-*"` requires `use_partial_assembly=True`.

## Local CUDA Stack Under `.local/`

This repository now includes a helper script that builds a local CUDA-enabled
stack under `Electrical/forward_modeling/.local/`:

```bash
source /home/wangyh/anaconda3/bin/activate torch

# Expected source trees. Reuse existing clones or create them first.
git clone https://github.com/hypre-space/hypre.git /tmp/hypre-cuda-src
git clone https://github.com/CEED/libCEED.git /tmp/libCEED-cuda-src
git clone https://github.com/mfem/mfem.git /tmp/mfem-cuda-src

GMESDataset/Electrical/forward_modeling/tools/build_cuda_stack.sh
```

The script builds:

- CUDA-enabled hypre
- CUDA-enabled libCEED
- CUDA-enabled MFEM with `MFEM_USE_CUDA=YES` and `MFEM_USE_CEED=YES`
- the local `mt_forward_mfem` module against that stack

For the local MFEM CUDA build the script uses a safer
`CMAKE_CUDA_FLAGS_RELEASE='-O1 -DNDEBUG'`, because `-O3` was enough to trigger
OOM on this workstation while compiling some large MFEM CUDA translation units.
It also applies local low-memory MFEM patches before configuring MFEM:

- `DiffusionIntegrator::AssembleEA` uses MFEM's generic EA diffusion kernel
  instead of fixed-order CUDA specializations.
- H(div) mass apply and H(div) quadrature interpolation keep their fallback
  implementations but skip large fixed-order CUDA specialization registration.
- Batched LOR assembly is compiled as an unsupported stub. This optional MFEM
  optimization is not used by the MT H(curl) forward path.

It uses the following local prefixes by default:

- `Electrical/forward_modeling/.local/hypre-cuda`
- `Electrical/forward_modeling/.local/libCEED-cuda`
- `Electrical/forward_modeling/.local/mfem-cuda`

The default local assumptions match the current workstation:

- CUDA toolkit: `/usr/local/cuda-12.8`
- host compiler: `/usr/bin/gcc-11`, `/usr/bin/g++-11`
- CUDA arch for MFEM/hypre: `120`
- CUDA arch for libCEED: `sm_120`

If you prefer to drive CMake manually, point this backend at the local MFEM
install with:

```bash
cmake -S GMESDataset/Electrical/forward_modeling -B /tmp/gmes_forward_modeling_build_cuda \
  "-DCMAKE_PREFIX_PATH=/home/wangyh/Project/GMESUni/GMESDataset/Electrical/forward_modeling/.local/mfem-cuda;$CONDA_PREFIX" \
  -DMFEM_DIR=/home/wangyh/Project/GMESUni/GMESDataset/Electrical/forward_modeling/.local/mfem-cuda/lib/cmake/mfem \
  -DPython3_EXECUTABLE="$CONDA_PREFIX/bin/python" \
  -Dpybind11_DIR="$CONDA_PREFIX/share/cmake/pybind11"
cmake --build /tmp/gmes_forward_modeling_build_cuda -j
```

If MFEM is not installed, Python-side interface tests can still run because they
mock the backend contract.
