import numpy as np
import os
import sys
import cigvis
import segyio

from core.builder import DatasetBuilder
from core.anomalies import EllipsoidAnomaly, SaltDomeAnomaly
# Import the new refactored classes
from core.anomalies.igneous_intrusion import IgneousIntrusion, IgneousIntrusionParams

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
    # Try to read dt
    try:
        dt = segyio.tools.dt(f) / 1000.0
        dz = dt if dt > 0 else 1.0
    except:
        dz = 1.0
        
    dx, dy = 1.0, 1.0 
    return vol, (dx, dy, dz)

def main():
    print("=== GMESDataset Anomaly Injection Demo ===")

    # Configuration for external SEGY files
    vp_segy_path = "/home/wangyh/DATAFOLDER/3DSeismic/AYLModel/3DExample/Velocity_choas/braided/AYL-00000.sgy"
    label_segy_path = "/home/wangyh/DATAFOLDER/3DSeismic/AYLModel/3DExample/Layer_choas/braided/AYL-00000.sgy"
    target_layer_label = 3
    
    use_segy = segyio is not None and os.path.exists(vp_segy_path) and os.path.exists(label_segy_path)

    if use_segy:
        print(f"1. Loading Background Vp from {vp_segy_path}...")
        vp_bg, _ = read_segy_volume(vp_segy_path)
        # Override geometry as requested: dx=dy=10m, dz=25m
        dx, dy, dz = 10.0, 10.0, 25.0
        
        nx, ny, nz = vp_bg.shape
        print(f"   Loaded volume: {nx}x{ny}x{nz}, range {vp_bg.min():.1f}-{vp_bg.max():.1f}")
        print(f"   Using geometry: dx={dx}m, dy={dy}m, dz={dz}m (Model Size: {nx*dx:.0f}x{ny*dy:.0f}x{nz*dz:.0f} m)")
        
        print(f"   Loading Label Model from {label_segy_path}...")
        label_vol, _ = read_segy_volume(label_segy_path)
        print(f"   Labels present in input: {np.unique(label_vol)}")
        
        if label_vol.shape != vp_bg.shape:
             print("Warning: Label volume shape mismatch! Assuming aligned for demo.")
    else:
        print("1. Creating synthetic background model (No SEGY files found)...")
        # Grid: 300 x 300 x 200
        nx, ny, nz = 300, 300, 200
        dx, dy, dz = 10.0, 10.0, 25.0 # dx=dy=10m, dz=25m
        
        # Simple linear gradient background
        vp_bg = np.zeros((nx, ny, nz), dtype=np.float32)
        for k in range(nz):
            vp_bg[:, :, k] = 2000.0 + k * dz * 0.5
            
        # Synthetic labels for testing stratigraphic logic
        label_vol = np.zeros((nx, ny, nz), dtype=np.int32)
        # Create a simple layered model (Layer 0, 1, 2)
        # Layer 1: between z=800 and z=1200
        z_idx_start = int(800.0 / dz)
        z_idx_end = int(1200.0 / dz)
        label_vol[:, :, z_idx_start:z_idx_end] = target_layer_label 

    print(f"   Background Vp range: {vp_bg.min():.1f} - {vp_bg.max():.1f} m/s")

    # 2. Initialize the Builder
    builder = DatasetBuilder(dx, dy, dz)

    # 3. Define Anomalies
    anomalies = []
    print("2. Defining anomalies...")

    # # -- Anomaly B: Salt Dome --
    # salt_params = SaltDomeAnomaly.create_random_params(
    #     grid_shape=(nx, ny, nz),
    #     grid_spacing=(dx, dy, dz),
    #     seed=12345
    # )
    # salt_params.z_top_m = 400.0
    # salt_params.mid_radius_m = 300.0
    # anom_dome = SaltDomeAnomaly(
    #     type="salt_dome",
    #     strength=0.0,
    #     edge_width_m=salt_params.edge_width_m,
    #     params=salt_params
    # )
    # anomalies.append(anom_dome) # Label 2

    # # -- Anomaly C: Igneous Dyke (Planar) --
    # print(f"   Adding Igneous Dyke...")
    # dyke_params = IgneousIntrusionParams(
    #     kind="dyke",
    #     cx_m=2200.0, cy_m=2200.0, cz_m=800.0,
    #     length_m=1200.0, width_m=800.0, max_thickness_m=40.0,
    #     strike_deg=45.0, dip_deg=80.0,
    #     roughness_amp_m=8.0
    # )
    # anom_dyke = IgneousIntrusion(
    #     type="dyke",
    #     strength=0.0, # Not used for igneous (absolute override)
    #     edge_width_m=dyke_params.edge_width_m,
    #     params=dyke_params
    # )
    # anomalies.append(anom_dyke) # Label 3

    # -- Anomaly D: Stratigraphic Sill (Controlled by Layer) --
    print(f"   Adding Stratigraphic Sill in Label {target_layer_label}...")
    # NOTE: We pass 'layer_labels' to the constructor so the logic can see the 3D volume
    sill_params = IgneousIntrusionParams(
        kind="sill",
        sill_layer_id=target_layer_label, # Use specific layer
        sill_alpha=0.5,                   # Middle of the layer
        sill_thick_max_frac_of_layer=0.4, # Occupy up to 40% of layer thickness
        length_m=2000.0,
        width_m=2000.0,
        max_thickness_m=80.0,
        sill_undulation_amp_m=20.0,
        # Center is approximate, logic will refine Z
        cx_m=1500.0, cy_m=1500.0
    )
    
    anom_sill = IgneousIntrusion(
        type="sill_layer",
        strength=0.0,
        edge_width_m=sill_params.edge_width_m,
        params=sill_params,
        layer_labels=label_vol # Pass the full 3D label volume
    )
    anomalies.append(anom_sill) # Label 4
    
    
    # 4. Inject Anomalies
    print("3. Injecting anomalies into model...")
    # Note: builder.inject_anomalies calls apply_to_vp(vp, X, Y, Z) on each anomaly.
    # IgneousIntrusion.apply_to_vp uses self.params and self.layer_labels
    vp_final, label, X, Y, Z = builder.inject_anomalies(vp_bg, anomalies)

    # 5. Verify Results
    print("4. Result Summary:")
    print(f"   Original Vp mean: {np.mean(vp_bg):.2f}")
    print(f"   Final Vp mean:    {np.mean(vp_final):.2f}")
    
    unique_labels = np.unique(label)
    print(f"   Found labels: {unique_labels}")
    
    print("5. Visualizing...")
    nodes = cigvis.create_slices(vp_final)
    
    # Add bodies
    # Anomaly order: 
    # 1: salt (ellipsoid)
    # 2: salt dome
    # 3: dyke
    # 4: sill
    
    colors = ['cyan', 'yellow', 'red', 'orange', 'purple']
    
    for i, lbl in enumerate(unique_labels):
        if lbl <= 0: continue
        
        # Determine color
        idx = int(lbl) - 1
        if idx < len(colors):
            c = colors[idx]
        else:
            c = 'gray'
            
        mask = (label == lbl).astype(np.float32)
        if np.any(mask):
            print(f"   Render Label {lbl} as {c}")
            nodes += cigvis.create_bodys(mask, level=0.5, color=c)

    cigvis.plot3D(nodes, xyz_axis=True)
    print("=== Done ===")

if __name__ == "__main__":
    main()
