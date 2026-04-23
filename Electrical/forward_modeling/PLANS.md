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
Create the MFEM-backed structure and compatibility layer while keeping the public Python API stable.

Context:
The legacy CUDA implementation is kept under `legacy_cuda/` for reference and fallback. The first MFEM implementation uses assembled operators and the scaled form `Kcurl + i * omega * mu0 * sigma * M`.

Constraints:
Do not add new CUDA kernels. Keep partial assembly behind configuration only. Do not make `mt_forward.py` require a built backend at import time.

Done when:
The new directory tree exists, the CMake configuration is defined, the Python facade prefers `mt_forward_mfem`, and local tests that do not require MFEM pass.
