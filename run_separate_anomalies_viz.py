import os
import sys
import numpy as np
import segyio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import cigvis

from core.builder import DatasetBuilder
from core.anomalies.igneous_intrusion import IgneousIntrusion, IgneousIntrusionParams
from core.anomalies.massive_sulfide import MassiveSulfide, MassiveSulfideParams
from core.anomalies.hydrocarbon_hydrate import HydrocarbonHydrate, HydrocarbonHydrateParams
from core.anomalies.brine_fault_zone import BrineFaultZone, BrineFaultZoneParams
from core.anomalies.sediment_basement_interface import SedimentBasementInterface, SedimentBasementParams
from core.anomalies.serpentinized_zone import SerpentinizedZone, SerpentinizedZoneParams
from core.anomalies.salt_dome_anomaly import SaltDomeAnomaly, SaltDomeParams

def read_segy_volume(path):
    print(f"Reading SEGY: {path}")
    with segyio.open(path, iline=segyio.TraceField.INLINE_3D, xline=segyio.TraceField.CROSSLINE_3D) as f:
        try:
            vol = segyio.tools.cube(f)
        except Exception:
            f.mmap()
            vol = segyio.tools.cube(f)
    dx, dy, dz = 10.0, 10.0, 25.0
    return vol, (dx, dy, dz)


def generate_and_plot(anomaly, name_en, name_zh, vp_bg, label_vol, dx, dy, dz, run_app=False):
    builder = DatasetBuilder(dx, dy, dz)
    print(f"\n-> Generating {name_zh}...")
    
    # Inject individually
    vp_final, mask_final, X, Y, Z = builder.inject_anomalies(vp_bg, [anomaly])
    
    save_dir = "/home/wangyh/Project/GMESUni/GMESDataset/DATAFOLDER/Cache/"
    os.makedirs(save_dir, exist_ok=True)
    
    nodes1 = cigvis.create_slices(vp_bg, cmap='jet')
    nodes2 = cigvis.create_slices(vp_final, cmap='jet')
    
    sub_labels = None
    try:
        # Calculate for different subclasses dynamically based on run_demo.py logic
        if hasattr(anomaly, "subtype_labels"):
            sub_labels = anomaly.subtype_labels(X, Y, Z)
            if anomaly.type == "brine_fault_zone":
                # Shift brine to avoid overlap for single viz volume
                viz_sub = np.zeros_like(sub_labels)
                viz_sub[(sub_labels >= 1) & (sub_labels < 10)] = 21 # All cores
                viz_sub[sub_labels >= 10] = 22 # All damage zones
                sub_labels = viz_sub
        elif hasattr(anomaly, "build_property_models"):
            multiprops = anomaly.build_property_models(X, Y, Z, vp_bg=vp_bg)
            if anomaly.type == "sediment_basement_interface":
                facies_sbi = multiprops.get("facies_label", None)
                if facies_sbi is not None:
                    viz_sub = np.zeros_like(facies_sbi)
                    viz_sub[facies_sbi == 2] = 30
                    viz_sub[facies_sbi == 3] = 31
                    sub_labels = viz_sub
            elif anomaly.type == "serpentinized_zone":
                sub_labels = multiprops.get("subtype", None)
        else:
            sub_labels = mask_final
    except Exception as e:
        print(f"Warning: Could not extract specific subtypes: {e}")
        sub_labels = mask_final

    if sub_labels is None:
        sub_labels = mask_final

    sub_colors = {
        1: 'red',      # GasLens / Core
        2: 'yellow',   # GasChimney
        3: 'cyan',     # HydrateLayer
        4: 'orange',   # FreeGasBelow
        5: 'gray',     # Halo 
        21: 'magenta', # BrineCore
        22: 'purple',  # BrineDamage
        30: 'brown',   # Basement
        31: 'green',   # Interface
        40: 'blue',    # SerpCore
        41: 'teal'     # SerpHalo
    }
    
    # To plot anomaly masks we can add it to the nodes
    if sub_labels is not None and len(np.unique(sub_labels)) > 1:
        for code in np.unique(sub_labels):
            if code == 0: continue
            
            c = sub_colors.get(code, 'magenta') # Default magenta
            sub_mask = (sub_labels == code).astype(np.float32)
            
            mask_nodes = cigvis.create_bodys(sub_mask, level=0.5, color=c, **{"alpha": 0.3})
            if isinstance(mask_nodes, list):
                nodes2.extend(mask_nodes)
            else:
                nodes2.append(mask_nodes)
            
    # Draw two side-by-side canvases and save
    out_file = f"Anomaly_{name_en}-{name_zh}.png"
    out_path = os.path.join(save_dir, out_file)
    cigvis.plot3D([nodes1, nodes2], grid=(1, 2), savename=out_file, savedir=save_dir, run_app=run_app, title=["原始速度模型", name_zh])
    
    print(f"Saved: {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_app", action="store_true", help="Launch cigvis GUI for checking visuals interactively.")
    args = parser.parse_args()

    vp_segy_path = "/home/wangyh/DATAFOLDER/3DSeismic/AYLModel/3DExample/Velocity_choas/braided/AYL-00000.sgy"
    label_segy_path = "/home/wangyh/DATAFOLDER/3DSeismic/AYLModel/3DExample/Layer_choas/braided/AYL-00000.sgy"
    
    vp_bg, (dx, dy, dz) = read_segy_volume(vp_segy_path)
    label_vol, _ = read_segy_volume(label_segy_path)
    nx, ny, nz = vp_bg.shape
    print(f'velocity shape = {nx,ny,nz}, dx={dx}, dy={dy}, dz={dz}')

    # 1. Dyke Swarm
    swarm_params = IgneousIntrusionParams(
        kind="swarm", dyke_x0_m=1000.0, dyke_y0_m=1500.0, dyke_z0_m=2500.0,
        dyke_thickness_m=40.0, dyke_length_m=2500.0, dyke_width_m=3000.0,
        dyke_strike_deg=45.0, dyke_dip_deg=80.0, swarm_count=5, swarm_spacing_m=300.0,
        swarm_fan_deg=15.0, vp_intr_mps=5000.0, aureole_enable=True
    )
    generate_and_plot(IgneousIntrusion(params=swarm_params, layer_labels=None, rng_seed=101), "Igneous_Swarm", "岩墙群", vp_bg, label_vol, dx, dy, dz, run_app=args.run_app)

    # 2. Stock (Plug)
    stock_params = IgneousIntrusionParams(
        kind="stock", stock_xc_m=2200.0, stock_yc_m=2200.0, stock_z_top_m=1500.0, stock_z_base_m=4500.0,
        stock_radius_m=400.0, vp_intr_mps=5800.0, aureole_enable=True
    )
    generate_and_plot(IgneousIntrusion(params=stock_params, layer_labels=None, rng_seed=202), "Igneous_Stock", "岩株", vp_bg, label_vol, dx, dy, dz, run_app=args.run_app)

    # 3. Gas Reservoir
    gas_params = HydrocarbonHydrateParams(
        kind="gas", layer_id=-1, center_x_m=1500.0, center_y_m=1500.0,
        lens_extent_x_m=1200.0, lens_extent_y_m=700.0, lens_thickness_m=120.0, vp_gas_mps=1800.0,
        gas_enable_chimney=True, chimney_height_m=1200.0, rng_seed=11
    )
    generate_and_plot(HydrocarbonHydrate(params=gas_params, layer_labels=label_vol), "Hydrocarbon_Gas", "气藏", vp_bg, label_vol, dx, dy, dz, run_app=args.run_app)

    # 4. Gas Hydrate
    hyd_params = HydrocarbonHydrateParams(
        kind="hydrate", layer_id=-1, center_x_m=2000.0, center_y_m=2000.0,
        hydrate_offset_above_m=40.0, hydrate_thickness_m=70.0, vp_hydrate_mps=3800.0,
        hard_gate_to_layer=True, hydrate_enable_patchy=False, rng_seed=22
    )
    generate_and_plot(HydrocarbonHydrate(params=hyd_params, layer_labels=label_vol), "Hydrocarbon_Hydrate", "天然气水合物", vp_bg, label_vol, dx, dy, dz, run_app=args.run_app)

    # 5. Brine Fault Zone
    brine_params = BrineFaultZoneParams(top_k=2, fault_quantile=0.996, core_thickness_m=40.0)
    generate_and_plot(BrineFaultZone(params=brine_params, vp_ref=vp_bg, rng_seed=999), "Brine_Fault", "含卤水断层", vp_bg, label_vol, dx, dy, dz, run_app=args.run_app)

    # 6. Sediment-Basement Interface
    sbi_params = SedimentBasementParams(
        use_layer_labels=True, basement_layer_id=-2, vp_basement_mps=6200.0, rng_seed=777
    )
    generate_and_plot(SedimentBasementInterface(params=sbi_params, layer_labels=label_vol), "Sediment_Basement", "沉积基底界面", vp_bg, label_vol, dx, dy, dz, run_app=args.run_app)

    # 7. Serpentinized Zone
    serp_params = SerpentinizedZoneParams(
        mode="patchy", use_layer_labels=True, corridor_length_m=2200.0, rng_seed=1212
    )
    generate_and_plot(SerpentinizedZone(params=serp_params, layer_labels=label_vol), "Serpentinized_Zone", "蛇纹岩化带", vp_bg, label_vol, dx, dy, dz, run_app=args.run_app)

    # 8. Massive Sulfide
    mass_sulfide_params = MassiveSulfideParams(
        layer_id=-1, center_x_m=1000.0, center_y_m=1000.0, lens_extent_x_m=600.0, lens_extent_y_m=500.0,
        lens_thickness_m=150.0
    )
    generate_and_plot(MassiveSulfide(params=mass_sulfide_params, layer_labels=label_vol), "Massive_Sulfide", "块状硫化物", vp_bg, label_vol, dx, dy, dz, run_app=args.run_app)

    # 9. Salt Dome
    salt_params = SaltDomeAnomaly.create_random_params((nx, ny, nz), (dx, dy, dz), seed=333)
    salt_dome = SaltDomeAnomaly(type="salt_dome", strength=0.0, edge_width_m=20.0, params=salt_params, rng_seed=333)
    generate_and_plot(salt_dome, "Salt_Dome", "盐丘", vp_bg, label_vol, dx, dy, dz, run_app=args.run_app)


if __name__ == '__main__':
    main()
