# GMESDataset: Geological Multi-physics Earth Simulation Dataset

[中文版本 (Chinese Version)](./README_CN.md)

**GMESDataset** is a comprehensive framework for constructing synthetic geological models and performing joint forward modeling of multiple geophysical fields (Gravity, Magnetic, Electrical, and Seismic). It is designed to generate highly realistic, complex 3D earth models containing various geological anomalies and facies, serving as a benchmark for joint inversion, multi-modal learning, and geophysical interpretation.

## 🌟 Key Features

### 1. Advanced Geological Modeling (`core/`)
A procedural generation engine capable of creating complex geological structures with multi-physics property consistency:
*   **Igneous Intrusions**: 3D dyke swarms, sills, and stocks/plugs with thermal aureoles.
*   **Hydrocarbon Systems**: Gas reservoirs (lens/traps), gas chimneys, and gas hydrates (BSRs, patchy saturation).
*   **Fault Systems**: Brine/Water-bearing faults with core and damage zones.
*   **Mineralization**: Massive sulfide deposits (high density/conductivity).
*   **Basement/Structural**: Sediment-Basement interfaces and salt domes.
*   **Alteration**: Serpentinized zones coupled to deep faults (high magnetic susceptibility, low density/velocity).

### 2. Multi-Physics Property Generation
Automatically generates consistent 3D volumes for:
*   **Seismic**: P-wave velocity ($V_p$).
*   **Gravity**: Density ($\rho$).
*   **Magnetic**: Susceptibility ($\chi$).
*   **Electrical**: Resistivity ($\Omega\cdot m$).

### 3. Forward Modeling Engines
*   **Seismic**: High-performance 2D/3D wave propagation simulation based on `deepwave` (PyTorch).
*   **Gravity**: 3D gravity anomaly forward modeling.
*   **Magnetic**: 3D magnetic anomaly simulation (MATLAB/Python).

## 📂 Directory Structure

```plaintext
GMESDataset/
├── core/                       # Earth Model Generation Engine
│   ├── anomalies/              # Specific geological anomaly implementations
│   ├── builder.py              # Main model assembly logic
│   └── petrophysics/           # Rock physics relationships
├── configs/                    # Configuration files for anomalies
├── Seismic/                    # Seismic Forward Modeling
│   └── forward_modeling/       # Deepwave-based simulation scripts
├── Gravity/                    # Gravity Field Simulation
├── Magnetic/                   # Magnetic Field Simulation
└── run_demo.py                 # Main entry script to generate models
```

## 🚀 Quick Start

### 1. Prerequisites
The project relies on standard scientific Python stacks. Key dependencies:
*   `numpy`, `scipy`, `matplotlib`
*   `segyio` (for reading SEGY background models)
*   `cigvis` (recommended for 3D visualization)
*   `torch`, `deepwave` (for seismic simulation)

### 2. Generating an Earth Model
Run the demo script to create a synthetic earth model with multiple anomalies injected into a background.

```bash
python run_demo.py
```

This script will:
1.  Load a background model (or create a synthetic one).
2.  Inject defined anomalies (Dykes, Faults, Gas, etc.).
3.  Calculate multi-physics properties (Vp, Rho, Chi, Res).
4.  Visualize the result interactively (if `cigvis` is installed).

### 3. Running Seismic Simulation
Navigate to the seismic modeling directory to run wave propagation simulations on the generated models.

```bash
cd Seismic/forward_modeling
# Edit config.yaml to point to your generated model file
python main.py --config config.yaml
```
## 📊 Geophysical Property Signatures

The table below summarizes the typical physical property variations for implemented geological anomalies, derived from the codebase defaults and standard rock physics.

| Anomaly Type | Velocity ($V_p$) | Density ($\rho$) | Resistivity ($\Omega\cdot m$) | Susceptibility ($\chi$) |
| :--- | :--- | :--- | :--- | :--- |
| **Igneous Intrusion** | **High** (~5500 m/s) <br> *Aureole: +3%* | **High** | **High** | **High** <br> *(if magnetic)* |
| **Massive Sulfide** | **Very High** (~6200 m/s) | **High** | **Very Low** (Conductive) | **High** <br> *(if pyrrhotite-rich)* |
| **Gas Reservoir** | **Low** (Gas ~1800 m/s) | **Low** | **High** | **Low** (Diamagnetic) |
| **Gas Hydrate** | **High** (Hydrate ~3700 m/s) <br> *Free Gas Below: Low* | **Low** | **High** | **Low** |
| **Brine Fault Zone** | **Low/Neutral** <br> (0% to -4% if fractured) | **Neutral/Low** | **Very Low** (Conductive) <br> *(Core ~0.5 $\Omega\cdot m$)* | **Neutral** |
| **Serpentinized Zone** | **Low** (-25%) | **Low** (-12%) | **Variable** | **High** (+0.02 SI) <br> *(Magnetite production)* |
| **Salt Dome** | **High** (~4500-5500 m/s) | **Low** (~2100-2200 kg/m^3) | **Very High** | **Low** (Diamagnetic) |
| **Sediment** | Background Trend | Background Trend | Background Trend | Background Trend |

> **Note**: Properties can be customized via config files or directly in `core/anomalies/*.py`. Reference physics logic: [GPT-4 Context](https://chatgpt.com/s/t_6988301699188191aa2a68b0f53dbecf).
