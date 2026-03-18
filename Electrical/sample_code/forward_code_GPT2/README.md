# em3d_pytorch_v5

PyTorch research prototype for **3D frequency-domain EM / MT** on a Yee staggered grid.

## New in v5
- **More rigorous 1D layered MT primary solver**:
  - Computes depth-dependent primary plane-wave fields in a 1D layered background.
  - Uses those primary fields inside a 3D **primary-secondary** MT formulation.
- **Stronger PML**:
  - Adds a simple **complex coordinate stretching (SC-PML / UPML-style)** layer on top of the earlier sponge damping.
- **Stronger block preconditioner**:
  - Adds a **vertical line preconditioner** (z-line tridiagonal solve) that is more suitable for MT/FDEM than pure Jacobi.

## Main APIs
- `simulate_mt_primary_secondary_batch(...)`
- `solve_mt_primary_secondary_frequency(...)`
- `solve_mt_primary_1d_fields(...)`
- `vertical_line_precond(...)`

## Notes
This is still a **research prototype**, not a production-grade MT/CSEM engine. The 1D primary solver is much more rigorous than the older boundary-excitation approximation, but it is still embedded in a prototype 3D Yee-grid FDFD code with simplified boundary handling and prototype solvers.


## v6 upgrade: layered 1D MT primary by analytic / recursion solver

This version replaces the older global 1D finite-difference primary profile with a more standard layered-earth MT primary:

1. Compute per-layer propagation constant `k = sqrt(-i*omega*mu*sigma)`.
2. Compute per-layer intrinsic impedance `Z = -i*omega*mu / k`.
3. Recurse bottom-up to obtain the input impedance at the top of each layer.
4. Propagate `(E,H)` through each layer analytically using downgoing / upgoing exponentials.

Why this matters in a 3D MT code: the full 3D solution is still solved on the 3D grid, but we split

`total = primary(1D layered background) + secondary(3D lateral anomaly response)`

so the 1D primary provides a physically cleaner incident/background field and a better source term for the 3D secondary solve.

## v7 upgrade: SimPEG-like MT Survey / Source / Receiver / Data API

This version adds a high-level API inspired by SimPEG's `natural_source` workflow.

New objects in `em3d.nsem`:

- `Rx.Impedance(locations, orientation, component)`
- `Rx.Tipper(locations, orientation, component)`
- `Src.PlanewaveXYPrimary(receiver_list, frequency)`
- `Survey(source_list)`
- `Data(survey, dobs)`
- `Simulation3DPrimarySecondary(mesh, survey, sigma, sigmaPrimary, ...)`

This allows scripts that look much closer to SimPEG examples:

```python
from em3d import Grid3D, NSEM

receiver_list = [
    NSEM.Rx.Impedance(rx_loc, orientation='xy', component='real'),
    NSEM.Rx.Impedance(rx_loc, orientation='xy', component='imag'),
]
source_list = [NSEM.Src.PlanewaveXYPrimary(receiver_list, freq) for freq in freqs]
survey = NSEM.Survey(source_list)
sim = NSEM.Simulation3DPrimarySecondary(grid, survey=survey, sigma=sig, sigmaPrimary=sig_bg)
data = NSEM.Data(survey=survey, dobs=sim.dpred())
```

The plotting helpers `Data.plot_app_res(...)` and `Data.plot_app_phs(...)` are also included for a SimPEG-like user workflow.


## v10 merged solver upgrade

This merged build keeps the v7 MT main project and replaces the original solver with the v9 flexible solver upgrade:

- `gmres(...)` now dispatches to flexible right-preconditioned `fgmres(...)`
- `block_gmres(...)` now dispatches to flexible right-preconditioned `block_fgmres(...)`
- `make_flexible_combo_precond(...)` is available to compose `vertical_line_precond(...)`, `jacobi_precond_from_diag(...)`, etc.
- Existing v7 forward APIs remain source-compatible.


## Default solver path

The current default MT solver path uses the flexible Krylov stack in `em3d/solver.py`.

- `precond='flex'` (default): vertical-line preconditioner followed by a Jacobi sweep
- `precond='line'`: vertical-line only
- `precond='jacobi'`: Jacobi only
- `precond='none'`: no preconditioner

New example: `examples/run_mt_simpeg_style_flexible.py`


## Flexible MT demo with saved outputs

The example `examples/run_mt_simpeg_style_flexible.py` is now a full demo that saves:

- `survey_layout.png`: receiver / station layout
- `central_station_rho_phase.png`: apparent resistivity and phase curves at the central station
- `predicted_data.npy`: flattened predicted data vector
- `receiver_locations.npy`: survey locations
- `central_station_mt_curves.npz`: complex MT curves for the central station
- `metadata.json`: frequencies, station layout, and basic run metadata

Run it with:

```bash
python examples/run_mt_simpeg_style_flexible.py
```
