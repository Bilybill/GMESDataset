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
from mt_forward import MTForward3D, mfem_cuda_enabled

print(mfem_cuda_enabled())
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

## CUDA-Enabled MFEM

MFEM's CUDA backend is a build-time feature. Build or install MFEM with CUDA and
a matching CUDA-enabled hypre, then rebuild `mt_forward_mfem` against that
installation. With MFEM's CMake build this means enabling CUDA, for example:

```bash
cmake -S /path/to/mfem -B /path/to/mfem-build \
  -DMFEM_USE_MPI=YES \
  -DMFEM_USE_CUDA=YES \
  -DCUDA_ARCH=sm_86 \
  -DCMAKE_INSTALL_PREFIX=/path/to/mfem-cuda-install
cmake --build /path/to/mfem-build -j
cmake --install /path/to/mfem-build
```

Then rebuild this backend with `CMAKE_PREFIX_PATH` pointing to that CUDA-enabled
MFEM install.

If MFEM is not installed, Python-side interface tests can still run because they
mock the backend contract.
