# AGENTS.md

## Scope
Only work inside `GMESDataset/Electrical/forward_modeling` unless explicitly asked.

## Goal
Replace the custom CUDA MT backend with an MFEM-based backend while preserving the public Python API in `mt_forward.py`.

## Public Interface Contract
- Keep class name `MTForward3D`.
- `forward(rho_tensor)` still accepts a contiguous `(NX, NY, NZ)` resistivity tensor.
- Return `(app_res, phase)` with shape `(n_freqs, NX, NY, 2)`.
- Keep the last dimension order `(Zxy, Zyx)`.
- Return tensors on the same device as the input tensor.

## Architecture Rules
- Keep `mt_forward.py` as the only public Python facade.
- New backend module name: `mt_forward_mfem`.
- Use MFEM parallel classes: `ParMesh`, `ND_FECollection`, `ParFiniteElementSpace`,
  `ParSesquilinearForm`, `ParComplexGridFunction`, `HypreAMS`, `GMRESSolver`.
- Use the scaled form `Kcurl + i * omega * mu0 * sigma * M`.
- First version must use assembled operators.
- Partial assembly may exist only behind a flag and may remain TODO.
- Do not add new custom CUDA kernels.
- Move the old backend into `legacy_cuda/`; do not delete it until tests pass.

## Files To Create
- `CMakeLists.txt`
- `bindings/pybind_module.cpp`
- `include/*.hpp`
- `src/*.cpp`
- `apps/mt_forward_cli.cpp`
- `tests/test_halfspace_smoke.py`
- `tests/test_interface_compat.py`
- `tests/test_frequency_order.py`

## Validation
- Configure with CMake using `MFEM_DIR`.
- Build the Python module and CLI.
- Run `ctest --output-on-failure`.
- Run `pytest GMESDataset/Electrical/forward_modeling/tests -q`.
- Add a halfspace smoke test.
- Add an interface compatibility test.

## Done When
- The new file tree exists.
- The MFEM backend builds.
- `mt_forward.py` prefers `mt_forward_mfem` and can still fall back to the legacy backend.
- Smoke tests pass.
