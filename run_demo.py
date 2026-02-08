#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import sys
try:
    import cigvis
except ImportError:
    cigvis = None
    print("Warning: cigvis not found. Visualization will be skipped.")
try:
    import segyio
except ImportError:
    segyio = None
    print("Warning: segyio not found. Using synthetic background.")

from core.builder import DatasetBuilder
from core.anomalies.igneous_intrusion import IgneousIntrusion, IgneousIntrusionParams
from core.anomalies.massive_sulfide import MassiveSulfide, MassiveSulfideParams
from core.anomalies.hydrocarbon_hydrate import HydrocarbonHydrate, HydrocarbonHydrateParams
from core.anomalies.brine_fault_zone import BrineFaultZone, BrineFaultZoneParams
from core.anomalies.sediment_basement_interface import SedimentBasementInterface, SedimentBasementParams
from core.anomalies.serpentinized_zone import SerpentinizedZone, SerpentinizedZoneParams

def read_segy_volume(path):
    """
    Reads a SEGY file as a 3D volume.
    Returns: volume (nx, ny, nz), (dx, dy, dz)
    """
    if segyio is None:
        raise ImportError("segyio is required to read SEGY files.")
        
    with segyio.open(path, iline=segyio.TraceField.INLINE_3D, xline=segyio.TraceField.CROSSLINE_3D) as f:
        try:
            vol = segyio.tools.cube(f)
        except Exception as e:
            f.mmap()
            vol = segyio.tools.cube(f)
    
    # Placeholder for geometry reading (simplified)
    dx, dy, dz = 1.0, 1.0, 1.0
    return vol, (dx, dy, dz)

def main():
    print("=== GMESDataset Igneous Intrusion Demo (Swarm & Stock) ===")

    # Configuration for external SEGY files
    vp_segy_path = "/home/wangyh/DATAFOLDER/3DSeismic/AYLModel/3DExample/Velocity_choas/braided/AYL-00000.sgy"
    label_segy_path = "/home/wangyh/DATAFOLDER/3DSeismic/AYLModel/3DExample/Layer_choas/braided/AYL-00000.sgy"
    
    use_segy = segyio is not None and os.path.exists(vp_segy_path) and os.path.exists(label_segy_path)

    if use_segy:
        print(f"1. Loading Background Vp from {vp_segy_path}...")
        vp_bg, _ = read_segy_volume(vp_segy_path)
        # Override geometry: dx=dy=10m, dz=25m
        dx, dy, dz = 10.0, 10.0, 25.0
        
        nx, ny, nz = vp_bg.shape
        print(f"   Loaded volume: {nx}x{ny}x{nz}, range {vp_bg.min():.1f}-{vp_bg.max():.1f}")
        print(f"   Using geometry: dx={dx}m, dy={dy}m, dz={dz}m (Model Size: {nx*dx:.0f}x{ny*dy:.0f}x{nz*dz:.0f} m)")
        
        print(f"   Loading Label Model from {label_segy_path}...")
        label_vol, _ = read_segy_volume(label_segy_path)
        
        if label_vol.shape != vp_bg.shape:
             print("Warning: Label volume shape mismatch! Assuming aligned for demo.")
    else:
        print("1. Creating synthetic background model (No SEGY files found)...")
        # Grid: 300 x 300 x 200
        nx, ny, nz = 300, 300, 200
        dx, dy, dz = 10.0, 10.0, 25.0 
        
        vp_bg = np.zeros((nx, ny, nz), dtype=np.float32)
        for k in range(nz):
            vp_bg[:, :, k] = 2000.0 + k * dz * 0.5
            
        label_vol = np.zeros((nx, ny, nz), dtype=np.int32)
        # Dummy labels (Layers 1, 2, 3) to ensure auto-pick always finds a valid layer
        label_vol[:, :, 40:80] = 1
        label_vol[:, :, 80:120] = 2
        label_vol[:, :, 120:160] = 3

    print(f"   Background Vp range: {vp_bg.min():.1f} - {vp_bg.max():.1f} m/s")

    # 2. Initialize the Builder
    builder = DatasetBuilder(dx, dy, dz)

    # 3. Define Anomalies
    anomalies = []
    print("2. Defining anomalies...")

    # --- Anomaly 1: Dyke Swarm ---
    # print("   -> Adding Dyke Swarm...")
    # swarm_params = IgneousIntrusionParams(
    #     kind="swarm",
    #     dyke_x0_m = 1000.0,
    #     dyke_y0_m = 1500.0,
    #     dyke_z0_m = 2500.0,    # Center depth
    #     dyke_thickness_m = 40.0,
    #     dyke_length_m = 2500.0, # Along strike
    #     dyke_width_m = 3000.0,  # Along dip (vertical extent approx)
    #     dyke_strike_deg = 45.0,
    #     dyke_dip_deg = 80.0,
    #     swarm_count = 5,
    #     swarm_spacing_m = 300.0,
    #     swarm_fan_deg = 15.0,   # Fan out slightly
    #     vp_intr_mps = 5000.0,
    #     aureole_enable = True
    # )
    # # Note: Swarm doesn't need layer_labels, but we pass it anyway or None
    # anom_swarm = IgneousIntrusion(params=swarm_params, layer_labels=None, rng_seed=101)
    # anomalies.append(anom_swarm)

    # # --- Anomaly 2: Stock (Plug) ---
    # print("   -> Adding Stock (Plug)...")
    # stock_params = IgneousIntrusionParams(
    #     kind="stock",
    #     stock_xc_m = 2200.0,
    #     stock_yc_m = 2200.0,
    #     stock_z_top_m = 1500.0,
    #     stock_z_base_m = 4500.0,
    #     stock_radius_m = 400.0, # Base radius
    #     stock_drift_m = 150.0,  # Sinuosity
    #     stock_roughness_eps = 0.15,
    #     stock_azimuth_power = 1.5,
    #     vp_intr_mps = 5800.0,
    #     aureole_enable = True,
    #     aureole_thickness_m = 120.0
    # )
    # anom_stock = IgneousIntrusion(params=stock_params, layer_labels=None, rng_seed=202)
    # anomalies.append(anom_stock)

    # --- Anomaly 3: Hydrocarbon Gas & Hydrate ---
    print("   -> Adding Gas Reservoir & Hydrate...")

    # --- Gas reservoir (trap + chimney) ---
    gas_params = HydrocarbonHydrateParams(
        kind="gas",
        layer_id=-1,                 # or set a specific layer id
        center_x_m=1500.0,
        center_y_m=1500.0,
        lens_extent_x_m=1200.0,
        lens_extent_y_m=700.0,
        lens_thickness_m=120.0,
        vp_gas_mps=1800.0,
        gas_enable_chimney=True,
        chimney_height_m=1200.0,
        chimney_radius_top_m=60.0,
        chimney_radius_base_m=160.0,
        rng_seed=11
    )
    # Pass label_vol if meaningful, else None. 
    # If label_vol is None, layer_id must be ignored or handled by geometric fallback.
    anom_gas = HydrocarbonHydrate(params=gas_params, layer_labels=label_vol)
    anomalies.append(anom_gas)

    # --- Gas hydrate (hydrate above + free gas below) ---
    hyd_params = HydrocarbonHydrateParams(
        kind="hydrate",
        layer_id=-1,
        center_x_m=2000.0, # Shifted position to not overlap perfectly
        center_y_m=2000.0,
        hydrate_offset_above_m=40.0,
        hydrate_thickness_m=70.0,
        vp_hydrate_mps=3800.0,
        hard_gate_to_layer=True,
        # GPT suggested temporarily disabling patchy to ensure visibility first, 
        # but since we applied the code fix, we can try keeping it True or follow suggestion A strictly.
        # User said "Strictly follow GPT". GPT says "Solution A ... hydrate_enable_patchy=False (to make sure visible)".
        hydrate_enable_patchy=False, 
        hydrate_patch_threshold=0.15,
        hydrate_enable_free_gas_below=True,
        vp_free_gas_mps=2000.0,
        rng_seed=22
    )
    anom_hyd = HydrocarbonHydrate(params=hyd_params, layer_labels=label_vol)
    anomalies.append(anom_hyd)

    # --- Anomaly 4: Brine / Water-bearing Fault Zone ---
    print("   -> Adding Brine/Water Fault Zone...")
    brine_params = BrineFaultZoneParams(
        top_k=2, 
        fault_quantile=0.996,             # High threshold to pick only major faults
        core_thickness_m=40.0,
        damage_width_m=200.0,
        rho_core_ohmm=0.5,                # Very conductive core
        rho_damage_factor=0.3,
        vp_delta_frac_in_core=-0.10,      # Decrease Vp by 10% in core (visible in Vp)
        vp_delta_frac_in_damage=-0.03,    # Decrease Vp by 3% in damage zone
        patch_enable=True,
        patch_strength=0.8                # Strong segmentation
    )
    # Note: BrineFaultZone needs vp_bg to extract structure
    anom_brine = BrineFaultZone(params=brine_params, vp_ref=vp_bg, rng_seed=999)
    # We append it to anomalies so builder can invoke apply_to_vp
    anomalies.append(anom_brine)

    # --- Anomaly 5: Sediment-Basement Interface ---
    print("   -> Adding Sediment-Basement Interface...")
    sbi_params = SedimentBasementParams(
        use_layer_labels=True,      # Use label_vol if available
        basement_layer_id=-2,       
        apply_mode="blend",
        vp_basement_mps=6200.0,
        edge_width_m=50.0,
        interface_band_m=80.0,      # New param in GPT version

        # Physics (Exponential compaction)
        rho_basement_gcc=2.85,      # Updated param name
        chi_basement_SI=0.05,       # Updated param name
        
        # Explicit compaction trends (optional overrides)
        vp_sed0_mps=1700.0,
        vp_sed_inf_mps=4000.0,
        rho_sed0_gcc=2.0,
        rho_sed_inf_gcc=2.6,
        
        # Seed is inside params in GPT version
        rng_seed=777
    )
    sbi = SedimentBasementInterface(params=sbi_params, layer_labels=label_vol)
    anomalies.append(sbi)

    # --- Anomaly 6: Serpentinized Zone ---
    print("   -> Adding Serpentinized Zone...")
    serp_params = SerpentinizedZoneParams(
        mode="patchy",
        use_layer_labels=True,
        fault_topk=2,
        layer_id=-1,
        corridor_length_m=2200.0,
        corridor_halfwidth_m=300.0,
        vp_delta_frac=-0.30,       # Strong Vp low
        rho_delta_frac=-0.15,      # Density decrease
        chi_add_SI=0.03,           # Magnetic Susceptibility increase
        rng_seed=1212
    )
    serp = SerpentinizedZone(params=serp_params, layer_labels=label_vol)
    anomalies.append(serp)

    # Manual Rho Demo
    print("   [Demo] Generating Multi-Physics Models (Rho, Res, Chi) via SBI...")
    nx, ny, nz = vp_bg.shape
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    z = np.arange(nz) * dz
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # SBI can generate full property models
    # Note: build_property_models needs X,Y,Z. 
    # Since builder only updates Vp, we verify properties separately here.
    multi_props = sbi.build_property_models(X, Y, Z, vp_bg=vp_bg)
    rho_sbi = multi_props["rho_gcc"]
    res_sbi = multi_props["resist_ohmm"]
    chi_sbi = multi_props["chi_SI"]
    facies_sbi = multi_props["facies_label"]

    print(f"   SBI Properties Generated:")
    print(f"     Rho range: {rho_sbi.min():.2f} - {rho_sbi.max():.2f} g/cc")
    print(f"     Res range: {res_sbi.min():.2f} - {res_sbi.max():.2f} Ohm.m")
    print(f"     Chi range: {chi_sbi.min():.4f} - {chi_sbi.max():.4f} SI")

    # Combine Brine Fault Rho into the SBI background (simple overwrite for demo)
    # Real pipeline would merge them carefully.
    # rho_final = anom_brine.apply_to_resistivity(res_sbi, X, Y, Z) 
    # (Note: anom_brine logic expects Rho to be resistivity? Method name is apply_to_resistivity.
    #  Let's treat res_sbi as the resistivity background for brine fault)
    res_final = anom_brine.apply_to_resistivity(res_sbi, X, Y, Z)



    # # Generate grid for debug only
    # nx, ny, nz = vp_bg.shape
    # x = np.arange(nx) * dx
    # y = np.arange(ny) * dy
    # z = np.arange(nz) * dz
    # X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # pre = anom_hyd._precompute(X, Y, Z)  # 临时调试
    # print("hydrate picked layer_id =", pre["layer_id"])
    # if pre.get("anchor_ok") is not None:
    #     print("anchor_ok coverage =", pre["anchor_ok"].mean())
    # print("z_c min/max =", np.nanmin(pre["z_c"]), np.nanmax(pre["z_c"]))

    # 4. Inject Anomalies
    print("3. Injecting anomalies into Vp model...")
    vp_final, mask_final, _, _, _ = builder.inject_anomalies(vp_bg, anomalies)

    print(f"   Final Vp min/max: {vp_final.min():.1f} / {vp_final.max():.1f}")
    
    # 5. Visualization with CigVis
    print("4. Visualizing result...")
    
    # Prepare visualization
    nodes = []
    
    # Background Vp
    # create_slices returns a list of nodes
    try:
        nodes += cigvis.create_slices(
            vp_final,
            cmap='jet'
        )
        
    except Exception as e:
        print(f"Warning: Could not create slices: {e}")

    # --- Generate Subtype Labels for HC/Hydrate Anomaly ---
    print("   Generating fine-grained subtype labels for Hydrocarbon/Hydrate...")
    nx, ny, nz = vp_bg.shape
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    z = np.arange(nz) * dz
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # We have anom_gas and anom_hyd. Let's merge their labels for visualization.
    # Note: If they overlap, one will overwrite the other in this simple merge.
    
    # Calculate for Gas
    # 0 bg, 1 lens, 2 chimney, 3 hydrate, 4 freegas, 5 halo
    sub_gas = anom_gas.subtype_labels(X, Y, Z)
    
    # Calculate for Hydrate
    sub_hyd = anom_hyd.subtype_labels(X, Y, Z)
    
    # Calculate for Brine Fault
    sub_brine = anom_brine.subtype_labels(X, Y, Z) # 1..k (Core), 11.. (Damage)
    
    # Shift brine to avoid overlap for single viz volume: 
    # Core (1->21), Damage (11->22)
    sub_brine_viz = np.zeros_like(sub_brine)
    sub_brine_viz[(sub_brine >= 1) & (sub_brine < 10)] = 21 # All cores as one color
    sub_brine_viz[sub_brine >= 10] = 22 # All damage zones as one color

    # Calculate for SBI (Sediment-Basement)
    # facies: 1=Sediment, 2=Basement, 3=Interface
    # We map 2->30 (Basement), 3->31 (Interface), 1->0 (Background)
    facies_sbi = multi_props["facies_label"]
    sub_sbi_viz = np.zeros_like(facies_sbi)
    sub_sbi_viz[facies_sbi == 2] = 30
    sub_sbi_viz[facies_sbi == 3] = 31

    # Calculate for Serpentinized Zone
    # Default params in SerpentinizedZone: subtype_core=40, subtype_halo=41
    # We verify/extract them here
    multi_props_serp = serp.build_property_models(X, Y, Z, vp_bg=vp_bg)
    sub_serp = multi_props_serp["subtype"]

    # Merge for single volume interaction (Priority: Serpentinized > Brine > Hydrate > Gas > SBI)
    # SBI is large (basement), so let anomalies override it
    merged_sub = np.where(sub_sbi_viz > 0, sub_sbi_viz, np.zeros_like(sub_gas))
    merged_sub = np.where(sub_gas > 0, sub_gas, merged_sub)
    merged_sub = np.where(sub_hyd > 0, sub_hyd, merged_sub)
    merged_sub = np.where(sub_brine_viz > 0, sub_brine_viz, merged_sub)
    merged_sub = np.where(sub_serp > 0, sub_serp, merged_sub)

    try:
        # 1. Standard anomalies from mask_final (if any others were added)
        # Here we only added the HC ones, so mask_final will contain 1 and 2 (as we appended them)
        # But we want to see the Detailed Subtypes instead.
        pass

        # 2. Render Subtypes
        # 1-5: Hydrocarbon System
        # 21-22: Brine Fault System
        # 30-31: Basement System
        # 40-41: Serpentinized System
        
        sub_colors = {
            1: 'red',      # GasLens (透镜状气藏)
            2: 'yellow',   # GasChimney (气烟囱)
            3: 'cyan',     # HydrateLayer (水合物层)
            4: 'orange',   # FreeGasBelow (水合物下伏游离气)
            5: 'gray',     # Halo (蚀变晕)
            21: 'magenta', # BrineCore (含卤水断层核)
            22: 'purple',  # BrineDamage (断层破碎带)
            30: 'brown',   # Basement (基岩)
            31: 'green',   # Interface (沉积-基岩界面)
            40: 'blue',    # SerpCore (蛇纹岩化核心 - 高磁低密度)
            41: 'teal'     # SerpHalo (蛇纹岩化边缘)
        }
        sub_names = {
            1: 'GasLens', 
            2: 'GasChimney', 
            3: 'HydrateLayer', 
            4: 'FreeGasBelow',
            5: 'Halo',
            21: 'BrineCore',
            22: 'BrineDamage',
            30: 'Basement',
            31: 'Interface',
            40: 'SerpCore',
            41: 'SerpHalo'
        }

        uni_sub = np.unique(merged_sub)
        sub_gas = anom_gas.subtype_labels(X, Y, Z)
        sub_hyd = anom_hyd.subtype_labels(X, Y, Z)

        print("gas unique/count:", np.unique(sub_gas, return_counts=True))
        print("hyd unique/count:", np.unique(sub_hyd, return_counts=True))

        print(f"   Subtypes found in volume: {uni_sub}")
        
        for code, c in sub_colors.items():
            if code not in uni_sub: continue
            
            # (Filter removed to show all types)
            if code != 40 and code != 41: continue

            name = sub_names.get(code, f'Type{code}')
            print(f"   Render Subtype {code}: {name} ({c})...")
            
            sub_mask = (merged_sub == code).astype(np.float32)
            
            # Use color for rendering
            nodes += cigvis.create_bodys(sub_mask, level=0.5, color=c)
                
    except Exception as e:
        print(f"Warning: Could not create bodys: {e}")
        import traceback
        traceback.print_exc()

    # If running in environment with GUI
    try:
        cigvis.plot3D(nodes, xyz_axis=True)
    except Exception as e:
        print(f"   Could not launch interactive plot: {e}")

if __name__ == "__main__":
    main()
