# GMESDataset: Geological Multi-physics Earth Simulation Dataset

[中文版本](./README_CN.md)

**GMESDataset** is a synthetic earth-modeling framework for building 3D geological anomaly models and generating consistent gravity, magnetic, electrical, and seismic forward responses. The project is organized around a shared anomaly registry, a facies-aware multiphysics property builder, and four forward solvers, so the same geological target can be visualized and forward modeled in a reproducible way.

## Key Features

### 1. Registry-driven geological anomaly modeling
The codebase currently maintains a unified anomaly registry in [`core/presets.py`](./core/presets.py), which centralizes preset names, builders, and exposure rules for visualization and forward modeling.

Registered anomaly families include:
- `igneous_swarm`: dyke swarm intrusion with thermal aureole
- `igneous_stock`: stock / plug intrusion
- `gas`: hydrocarbon gas lens / chimney system
- `hydrate`: gas hydrate system with optional free gas below
- `brine_fault`: brine-bearing fault core and damage zone
- `massive_sulfide`: lens + chimney + stockwork sulfide mineralization
- `salt_dome`: salt dome structure
- `sediment_basement`: sediment-basement interface
- `serpentinized`: serpentinized alteration zone

### 2. Facies-aware multiphysics property generation
Starting from background velocity and layer-label volumes, GMESDataset generates aligned 3D volumes for:
- `vp`: P-wave velocity
- `rho`: density in `g/cm^3`
- `res`: resistivity in `Ohm-m`
- `chi`: magnetic susceptibility in SI

Background properties are generated with facies-aware rock-physics logic, then anomaly-specific perturbations are injected in a physics-consistent way.

### 3. Four forward modeling engines
- `Seismic`: 3D acoustic wave propagation based on `deepwave`
- `Gravity`: 3D gravity anomaly forward modeling
- `Magnetic`: 3D magnetic anomaly forward modeling
- `Electrical`: 3D MT forward modeling through the `Electrical/forward_modeling` extension

### 4. Unified data packaging
`run_multiphysics_forward.py` saves the generated 3D models, acquisition metadata, and all available forward responses into a single `forward_bundle.npz`, making downstream loading and paired analysis much easier.

## Directory Structure

```text
GMESDataset/
├── core/
│   ├── anomalies/                 # Geological anomaly implementations
│   ├── forward_modeling/          # Gravity / magnetic / MT / seismic wrappers
│   ├── petrophysics/              # Rock-physics transforms
│   ├── presets.py                 # Shared anomaly registry and SEGY helpers
│   ├── multiphysics.py            # Multiphysics property builder
│   └── viz_utils.py               # Shared visualization helpers
├── Electrical/                    # MT forward modeling extension and tests
├── Gravity/                       # Gravity forward modeling code
├── Magnetic/                      # Magnetic forward modeling code
├── Seismic/                       # Deepwave-based seismic modeling code
├── run_multiphysics_viz.py        # Visualize one anomaly in four properties
├── run_all_anomalies_viz.py       # Compare multiple registered anomalies
├── run_separate_anomalies_viz.py  # Save anomaly-by-anomaly visualizations
├── run_multiphysics_forward.py    # Joint gravity / magnetic / MT / seismic forward run
└── plot_saved_forward_data.py     # Reload and visualize saved outputs
```

## Quick Start

### 1. Dependencies
Core dependencies:
- `numpy`, `scipy`, `matplotlib`
- `torch`
- `segyio`
- `cigvis` for 3D visualization
- `deepwave` for seismic forward modeling

Optional but recommended:
- CUDA-capable PyTorch environment for seismic, gravity, magnetic, and MT preprocessing

### 2. Visualize a multiphysics anomaly model
Use the registry-backed visualization entry:

```bash
python run_multiphysics_viz.py
```

This builds a background model from the configured SEGY volumes, injects one registered anomaly preset, and visualizes `vp / rho / res / chi` together.

### 3. Run joint forward modeling
The main joint forward entry is:

```bash
python run_multiphysics_forward.py --device cuda
```

Useful options:
- `--anomaly-type {igneous_swarm,brine_fault,massive_sulfide,salt_dome}`
- `--seismic-preset {full,light}`
- `--seismic-batch-size N`
- `--skip_mt`
- `--skip_seismic`

Example:

```bash
python run_multiphysics_forward.py \
  --anomaly-type massive_sulfide \
  --device cuda \
  --seismic-preset light \
  --seismic-batch-size 8
```

### 4. Inspect saved outputs
After a successful run, all 3D models and forward responses are packed into:

```text
DATAFOLDER/Cache/ForwardOutput/forward_bundle.npz
```

The current bundle layout is documented in [`README_FORWARD_OUTPUTS.md`](./README_FORWARD_OUTPUTS.md).

## Geophysical Property Signature Table

The table below summarizes the current code defaults and the expected four-method response tendencies. The code is authoritative; representative values here are intentionally rounded for readability.

| Target / Preset | Velocity `Vp` | Density `rho` | Resistivity `res` | Susceptibility `chi` | Joint interpretation cue |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `igneous_swarm`, `igneous_stock` | High. Registry presets use about `5000-5800 m/s`; aureole adds roughly `+3%` locally. | High, typically around `3.0 g/cm^3`; aureole about `+2%`. | Very high, around `5000 Ohm-m`; aureole slightly decreases relative to the intrusion core. | High, around `0.05 SI`, with a mild aureole increase. | Strong seismic, gravity, magnetic, and MT contrast; a good balanced benchmark target. |
| `massive_sulfide` | Very high in the core, about `5200-6200 m/s`; halo slightly lower than background. | High, roughly `3.0-4.0 g/cm^3`. | Very low, about `0.5-10 Ohm-m`; one of the strongest conductive targets in the project. | High to very high, about `1.8e-3` to `1.2e-2 SI`, plus a weak magnetic halo. | Produces compact high-density, high-susceptibility, low-resistivity ore-style responses. |
| `gas` | Low, about `1800 m/s`. | Low, about `2.0 g/cm^3`. | High, about `100 Ohm-m`. | Low / near background. | Strong seismic low-velocity anomaly with weak magnetic response and resistive electrical signature. |
| `hydrate` | Hydrate layer high, about `3700 m/s`; optional free gas below about `2000 m/s`. | Moderate, about `2.3 g/cm^3` in hydrate and `2.1 g/cm^3` in free gas. | High in hydrate, about `200 Ohm-m`; lower in free gas, about `50 Ohm-m`. | Low / near background. | Distinct seismic layering and electrical contrast, but usually weak magnetic response. |
| `brine_fault` | Neutral by default in the current preset; code keeps fault-core and damage-zone velocity deltas at `0%` unless configured. | Near background by default. | Extremely low in the conductive core, about `0.5 Ohm-m`. | Near background. | Dominantly an MT target; gravity and magnetic signatures are weak unless you customize density or susceptibility. |
| `salt_dome` | High, randomly drawn around `4500-5500 m/s`. | Low, about `2.15 g/cm^3`. | Very high, about `3000 Ohm-m`. | Very low to slightly negative, about `-1e-5 SI`. | Classic fast, resistive, low-density, weakly magnetic salt body. |
| `sediment_basement` | Sediments trend from about `1700` to `4000 m/s`; basement is about `6200 m/s` in the preset. | Sediments trend from about `1.95` to `2.45 g/cm^3`; basement about `2.75 g/cm^3`. | Sediments trend from about `5` to `80 Ohm-m`; basement about `2000 Ohm-m`. | Sediments about `5e-4 SI`; basement about `0.02 SI`. | Good structural benchmark for broad multi-property contrasts across an interface. |
| `serpentinized` | Lower than background by about `25%`. | Lower than background by about `12%`. | Lower than background by about `30%`. | Higher than background by an absolute addition of about `+0.02 SI`. | Good alteration-style target: slower and lighter, but markedly more magnetic. |

## Notes

- `run_multiphysics_forward.py` currently exposes four turnkey forward presets through `--anomaly-type`: `igneous_swarm`, `brine_fault`, `massive_sulfide`, and `salt_dome`.
- The broader registry already supports additional presets for visualization and future forward extensions.
- If the README and code ever disagree, please treat the implementation under [`core/`](./core) and [`run_multiphysics_forward.py`](./run_multiphysics_forward.py) as the source of truth.
