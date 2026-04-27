# MFEM Migration Plan

1. Quarantine legacy backend into `legacy_cuda/`.
2. Add `CMakeLists.txt`, pybind skeleton, and new source tree.
3. Implement tensor hex mesh builder.
4. Implement tensor resistivity to conductivity coefficient.
5. Implement uniform halfspace boundary coefficients.
6. Implement assembled MFEM solve path for one frequency and one polarization.
7. Extend to both polarizations and full response extraction.
8. Update `mt_forward.py` compatibility wrapper.
9. Add smoke tests and interface regression tests.
10. Benchmark against legacy backend on a small model.

## Current Execution Slice

Goal:
Install a CUDA-enabled local MFEM stack under `forward_modeling/.local/`, rebuild
`mt_forward_mfem` against it, and expose the partial-assembly / `ceed-cuda`
execution path without changing the public Python API.

Context:
The legacy CUDA implementation remains under `legacy_cuda/` for reference and
fallback. The first MFEM implementation already uses the scaled form
`Kcurl + i * omega * mu0 * sigma * M`; the new execution slice extends it with
device-aware runtime selection, CUDA-enabled external dependencies, and a
matrix-free path for CEED-backed execution.

Constraints:
Do not add new custom CUDA kernels. Keep partial assembly behind an explicit
flag. Keep `mt_forward.py` importable when no backend is present. Preserve the
existing return shapes and frequency ordering.

Done when:
CUDA-enabled hypre/libCEED/MFEM install under `.local/`, `mt_forward_mfem`
rebuilds against that install, Python can request `device="cuda"` or
`device="ceed-cuda:/gpu/cuda/shared"` with `use_partial_assembly=True`, and the
assembled CPU path remains available for regression comparison.
