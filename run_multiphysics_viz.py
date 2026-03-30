import os
import sys
import numpy as np
import segyio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import cigvis

from core.multiphysics import build_multiphysics_model
from core.anomalies.igneous_intrusion import IgneousIntrusion, IgneousIntrusionParams
from core.anomalies.hydrocarbon_hydrate import HydrocarbonHydrate, HydrocarbonHydrateParams
from core.anomalies.brine_fault_zone import BrineFaultZone, BrineFaultZoneParams
from core.anomalies.sediment_basement_interface import SedimentBasementInterface, SedimentBasementParams
from core.anomalies.serpentinized_zone import SerpentinizedZone, SerpentinizedZoneParams
from core.anomalies.massive_sulfide import MassiveSulfide, MassiveSulfideParams
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

def generate_multiphysics_and_plot(anomaly, anomaly_type_str, name_en, name_zh, vp_bg, label_vol, dx, dy, dz, run_app=False, show_colorbar=True):
    """
    基于速度模型生成多物理场属性（密度，电阻率，磁化率），叠加异常体，并使用 cigvis 进行可视化。
    """
    print(f"\n-> Generating Multiphysics for {name_zh} ({anomaly_type_str})...")
    
    model = build_multiphysics_model(vp_bg, label_vol, [anomaly], dx, dy, dz)
    vp_multi = model["vp"]
    rho_multi = model["rho"]
    res_multi = model["resist"]
    chi_multi = model["chi"]
    mask_final = model["anomaly_label"]
    X, Y, Z = model["X"], model["Y"], model["Z"]
    
    # 保存与可视化准备
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, "DATAFOLDER", "Cache/", "Fig")
    os.makedirs(save_dir, exist_ok=True)
    
    # 磁化率放大数值便于可视化
    chi_multi_scaled = chi_multi * 1e5
    
    # 获取各个场的合理取值范围 (clim)，用于切片色标和表面物理场上色
    def _safe_clim(vmin, vmax, eps=1e-5):
        return [vmin, vmax] if vmax > vmin else [vmin, vmax + eps]

    clim_vp = _safe_clim(vp_multi.min(), vp_multi.max())
    clim_rho = _safe_clim(rho_multi.min(), rho_multi.max())
    
    log_res_multi = np.log10(np.clip(res_multi, a_min=1e-5, a_max=None))
    # 电阻率跨度极大，往往取对数以保证可视化效果
    clim_res = _safe_clim(log_res_multi.min(), log_res_multi.max())
    clim_chi = _safe_clim(chi_multi_scaled.min(), chi_multi_scaled.max())

    # 在四个子窗口分别创建模型基础切片
    nodes_vp = cigvis.create_slices(vp_multi, cmap='jet', clim=clim_vp)
    nodes_rho = cigvis.create_slices(rho_multi, cmap='jet', clim=clim_rho)
    nodes_res = cigvis.create_slices(log_res_multi, cmap='jet', clim=clim_res)
    nodes_chi = cigvis.create_slices(chi_multi_scaled, cmap='jet', clim=clim_chi)
    
    # ---------------- 异常体可视化 ----------------
    sub_labels = None
    try:
        if hasattr(anomaly, "subtype_labels"):
            sub_labels = anomaly.subtype_labels(X, Y, Z)
            if anomaly.type == "brine_fault_zone":
                viz_sub = np.zeros_like(sub_labels)
                viz_sub[(sub_labels >= 1) & (sub_labels < 10)] = 21 
                viz_sub[sub_labels >= 10] = 22 
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
    
    if sub_labels is not None and len(np.unique(sub_labels)) > 1:
        for code in np.unique(sub_labels):
            if code == 0: continue
            
            sub_mask = (sub_labels == code).astype(np.float32)
            
            for nds, vol, clim in zip(
                [nodes_vp, nodes_rho, nodes_res, nodes_chi],
                [vp_multi, rho_multi, log_res_multi, chi_multi_scaled],
                [clim_vp, clim_rho, clim_res, clim_chi]
            ):
                mask_nodes = cigvis.create_bodys(sub_mask, level=0.5, color='white', alpha=1.0)
                if mask_nodes:
                    import scipy.ndimage as ndimage
                    node = mask_nodes[0]
                    verts = node.mesh_data.get_vertices()
                    if verts is not None and len(verts) > 0:
                        # 采用三线性插值(order=1)以消除浮点坐标强转整数带来的莫尔条纹/锯齿
                        coords = verts.T  # map_coordinates expect shape (3, N)
                        vals = ndimage.map_coordinates(vol, coords, order=1, mode='nearest')
                        
                        norm = mcolors.Normalize(vmin=clim[0], vmax=clim[1])
                        scalar_map = cm.ScalarMappable(norm=norm, cmap='jet')
                        colors = scalar_map.to_rgba(vals, alpha=1.0)
                        
                        node.mesh_data.set_vertex_colors(colors)

                    if isinstance(mask_nodes, list):
                        nds.extend(mask_nodes)
                    else:
                        nds.append(mask_nodes)
    # ---------------- 异常体可视化结束 ----------------

    # 统一增加 colorbar
    if show_colorbar:
        cb_vp = cigvis.create_colorbar_from_nodes(nodes_vp, label_str="Vp (m/s)")
        cb_rho = cigvis.create_colorbar_from_nodes(nodes_rho, label_str="Density (g/cm^3)")
        cb_res = cigvis.create_colorbar_from_nodes(nodes_res, label_str="log10(Res)")
        cb_chi = cigvis.create_colorbar_from_nodes(nodes_chi, label_str="Suscept(1e-5SI)")
        
        for nds, cb in zip((nodes_vp, nodes_rho, nodes_res, nodes_chi), (cb_vp, cb_rho, cb_res, cb_chi)):
            if isinstance(cb, list):
                nds.extend(cb)
            else:
                nds.append(cb)
    
    out_file = f"Multiphysics_{name_en}-{name_zh}.png"
    out_path = os.path.join(save_dir, out_file)
    
    # (2, 2) 网格绘制四个物理场：Vp, Rho, Res(log), Chi
    cigvis.plot3D(
        [nodes_vp, nodes_rho, nodes_res, nodes_chi], 
        grid=(2, 2), 
        savename=out_file, 
        savedir=save_dir, 
        run_app=run_app, 
        title=[f"Vp (m/s) - {name_zh}", f"Density (g/cm^3)", f"log10(Resistivity) (Ohm.m)", f"Susceptibility (x1e-5 SI)"]
    )
    print(f"Saved Multiphysics Visualization: {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_app", action="store_true", help="Launch cigvis GUI for checking visuals interactively.")
    parser.add_argument("--hide_colorbar", action="store_true", help="Hide colorbars in the output visualization.")
    parser.add_argument("--vp_segy", type=str, default="/home/wangyh/DATAFOLDER/3DSeismic/AYLModel/3DExample/Velocity_choas/braided/AYL-00000.sgy", help="Path to background Vp SEGY.")
    parser.add_argument("--label_segy", type=str, default="/home/wangyh/DATAFOLDER/3DSeismic/AYLModel/3DExample/Layer_choas/braided/AYL-00000.sgy", help="Path to background label SEGY.")
    args = parser.parse_args()

    vp_segy_path = args.vp_segy
    label_segy_path = args.label_segy
    
    try:
        vp_bg, (dx, dy, dz) = read_segy_volume(vp_segy_path)
        label_vol, _ = read_segy_volume(label_segy_path)
        nx, ny, nz = vp_bg.shape
        print(f'Velocity shape = {nx, ny, nz}, dx={dx}, dy={dy}, dz={dz}')
    except Exception as e:
        raise RuntimeError(f"Failed to read SEGY volumes: {e}")

    # 1. Dyke Swarm -> Igneous
    swarm_params = IgneousIntrusionParams(
        kind="swarm", dyke_x0_m=1000.0, dyke_y0_m=1500.0, dyke_z0_m=2500.0,
        dyke_thickness_m=40.0, dyke_length_m=2500.0, dyke_width_m=3000.0,
        dyke_strike_deg=45.0, dyke_dip_deg=80.0, swarm_count=5, swarm_spacing_m=300.0,
        swarm_fan_deg=15.0, vp_intr_mps=5000.0, aureole_enable=True
    )
    generate_multiphysics_and_plot(
        IgneousIntrusion(params=swarm_params, layer_labels=None, rng_seed=101),
        "Igneous", "Igneous_Swarm", "岩墙群", vp_bg, label_vol, dx, dy, dz, run_app=args.run_app, show_colorbar=not args.hide_colorbar
    )
    
    # 2. Gas Reservoir -> Gas
    gas_params = HydrocarbonHydrateParams(
        kind="gas", layer_id=-1, center_x_m=1500.0, center_y_m=1500.0,
        lens_extent_x_m=1200.0, lens_extent_y_m=700.0, lens_thickness_m=120.0, vp_gas_mps=1800.0,
        gas_enable_chimney=True, chimney_height_m=1200.0, rng_seed=11
    )
    generate_multiphysics_and_plot(
        HydrocarbonHydrate(params=gas_params, layer_labels=label_vol),
        "Gas", "Hydrocarbon_Gas", "气藏", vp_bg, label_vol, dx, dy, dz, run_app=args.run_app, show_colorbar=not args.hide_colorbar
    )

    # 3. Gas Hydrate -> Hydrate
    hyd_params = HydrocarbonHydrateParams(
        kind="hydrate", layer_id=-1, center_x_m=2000.0, center_y_m=2000.0,
        hydrate_offset_above_m=40.0, hydrate_thickness_m=70.0, vp_hydrate_mps=3800.0,
        hard_gate_to_layer=True, hydrate_enable_patchy=False, rng_seed=22
    )
    generate_multiphysics_and_plot(
        HydrocarbonHydrate(params=hyd_params, layer_labels=label_vol),
        "Hydrate", "Hydrocarbon_Hydrate", "天然气水合物", vp_bg, label_vol, dx, dy, dz, run_app=args.run_app, show_colorbar=not args.hide_colorbar
    )

    # 4. Brine Fault Zone -> BrineFault
    brine_params = BrineFaultZoneParams(top_k=2, fault_quantile=0.996, core_thickness_m=40.0)
    generate_multiphysics_and_plot(
        BrineFaultZone(params=brine_params, vp_ref=vp_bg, rng_seed=999),
        "BrineFault", "Brine_Fault", "含卤水断层", vp_bg, label_vol, dx, dy, dz, run_app=args.run_app, show_colorbar=not args.hide_colorbar
    )

    # 5. Massive Sulfide -> Sulfide
    mass_sulfide_params = MassiveSulfideParams(
        layer_id=-1, center_x_m=1000.0, center_y_m=1000.0, lens_extent_x_m=600.0, lens_extent_y_m=500.0,
        lens_thickness_m=150.0
    )
    generate_multiphysics_and_plot(
        MassiveSulfide(params=mass_sulfide_params, layer_labels=label_vol),
        "Sulfide", "Massive_Sulfide", "块状硫化物", vp_bg, label_vol, dx, dy, dz, run_app=args.run_app, show_colorbar=not args.hide_colorbar
    )
    
    # 6. Salt Dome -> SaltDome
    salt_params = SaltDomeAnomaly.create_random_params((nx, ny, nz), (dx, dy, dz), seed=333)
    salt_dome = SaltDomeAnomaly(type="salt_dome", strength=0.0, edge_width_m=20.0, params=salt_params, rng_seed=333)
    generate_multiphysics_and_plot(
        salt_dome, "SaltDome", "Salt_Dome", "盐丘", vp_bg, label_vol, dx, dy, dz, run_app=args.run_app, show_colorbar=not args.hide_colorbar
    )
    
    # 7. Sediment Basement Interface -> Basement
    base_params = SedimentBasementParams()
    generate_multiphysics_and_plot(
        SedimentBasementInterface(params=base_params, layer_labels=label_vol),
        "Basement", "Sediment_Basement", "沉积-基底界面", vp_bg, label_vol, dx, dy, dz, run_app=args.run_app, show_colorbar=not args.hide_colorbar
    )

    # 8. Serpentinized Zone -> Serpentinized
    serp_params = SerpentinizedZoneParams()
    generate_multiphysics_and_plot(
        SerpentinizedZone(params=serp_params, layer_labels=label_vol),
        "Serpentinized", "Serpentinized_Zone", "蛇纹石化带", vp_bg, label_vol, dx, dy, dz, run_app=args.run_app, show_colorbar=not args.hide_colorbar
    )

if __name__ == '__main__':
    main()
