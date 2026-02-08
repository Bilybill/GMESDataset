import numpy as np
import os
import sys
# Ensure we can import from the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
# Add cigvis to path
sys.path.append("/home/wangyh/Project/cigvis")
import cigvis
try:
    from cigvis import vispyplot
    # Monkey patch if not automatically exposed (e.g. if _has_vispy check failed but we want to try anyway)
    if not hasattr(cigvis, 'create_slices'):
        cigvis.create_slices = vispyplot.create_slices
        cigvis.create_bodys = vispyplot.create_bodys
        cigvis.plot3D = vispyplot.plot3D
except ImportError as e:
    print(f"Warning: Could not import vispyplot: {e}")

# Optional: Try importing segyio
try:
    import segyio
except ImportError:
    segyio = None
    print("Warning: segyio not found. SEGY loading will be disabled.")

from core.builder import DatasetBuilder
from core.anomalies import EllipsoidAnomaly, SaltDomeAnomaly, IgneousIntrusion

def read_segy_volume(path):
    """
    Reads a SEGY file as a 3D volume.
    Returns: volume (nx, ny, nz), (dx, dy, dz)
    Assumes regular geometry.
    """
    if segyio is None:
        raise ImportError("segyio is required to read SEGY files.")
        
    with segyio.open(path, iline=segyio.tracefield.keys.INLINE_3D, xline=segyio.tracefield.keys.CROSSLINE_3D) as f:
        # Read as cube
        # segyio cube is usually (il, xl, samples)
        # We need to assume standard sorting
        try:
            vol = segyio.tools.cube(f)
        except Exception as e:
            # Fallback for some files that might explicitly need sorting
            # print(f"Direct cube read failed ({e}), trying strict mode...")
            f.mmap()
            vol = segyio.tools.cube(f)
            
        # Standardize strictly to (Inline, Crossline, Depth/Time) -> (X, Y, Z)
        # Dimensions
        n_il, n_xl, n_samples = vol.shape
        
        # Get deltas
        # dz (sample rate)
        dt = segyio.tools.dt(f) / 1000.0 # us to ms if using time, or just units
        # If headers are depth, dt might be depth step. Assuming meters if Z.
        # Fallback to 1.0 if not trustworthy or check headers.
        
        # dx, dy
        # Calculate from coordinates if possible, else 1.0
        # Simple placeholder logic
        dx, dy, dz = 1.0, 1.0, 1.0 
        
        # If we assume volume orientations are X, Y, Z
        # We prefer (nx, ny, nz)
        return vol, (dx, dy, dz)

def extract_horizon_from_label(label_vol, target_label, dz=1.0, mode='top'):
    """
    Extracts a 2D depth map from a 3D label volume.
    mode: 'top' (first occurrence) or 'bottom' (last occurrence)
    """
    nx, ny, nz = label_vol.shape
    depths = np.full((nx, ny), np.nan, dtype=np.float32)
    
    # Iterate columns (could be vectorized but loop is clearer for this logic)
    # Using numpy argmax for speed
    
    mask = (label_vol == target_label)
    
    # Check if target exists in column
    # any along axis 2
    exists = np.any(mask, axis=2)
    
    if mode == 'top':
         # np.argmax returns index of first True
         indices = np.argmax(mask, axis=2)
         # argmax returns 0 if all False, so we must use 'exists' mask
    else:
         # Flip along Z to find last
         indices = nz - 1 - np.argmax(mask[:, :, ::-1], axis=2)
    
    # Fill depths
    # Where exists is true, depth = index * dz
    good_indices = indices[exists]
    depths[exists] = good_indices * dz
    
    return depths

def main():
    print("=== GMESDataset Anomaly Injection Demo ===")

    # Configuration for external SEGY files (Set these to test)
    # If these files exist, the demo will use them instead of synthetic background
    vp_segy_path = "Model_mp.segy"  # Example path
    label_segy_path = "Label_mp.segy" # Example path
    target_layer_label = 3
    
    use_segy = segyio is not None and os.path.exists(vp_segy_path) and os.path.exists(label_segy_path)

    if use_segy:
        print(f"1. Loading Background Vp from {vp_segy_path}...")
        vp_bg, (dx, dy, dz) = read_segy_volume(vp_segy_path)
        nx, ny, nz = vp_bg.shape
        print(f"   Loaded volume: {nx}x{ny}x{nz}, range {vp_bg.min():.1f}-{vp_bg.max():.1f}")
        
        print(f"   Loading Label Model from {label_segy_path}...")
        label_vol, _ = read_segy_volume(label_segy_path)
        
        # Check consistency
        if label_vol.shape != vp_bg.shape:
             print("Warning: Label volume shape mismatch! Resizing or cropping might be needed in real app.")
             # For demo simplicity, we assume they match
    else:
        print("1. Creating synthetic background model (No SEGY files found or segyio missing)...")
        # Grid: 300 x 300 x 200
        nx, ny, nz = 300, 300, 200
        dx, dy, dz = 10.0, 10.0, 10.0  # meters
        
        # Simple linear gradient background: 2000 m/s at top, increasing with depth
        vp_bg = np.zeros((nx, ny, nz), dtype=np.float32)
        for k in range(nz):
            vp_bg[:, :, k] = 2000.0 + k * dz * 0.5
            
    print(f"   Background Vp range: {vp_bg.min():.1f} - {vp_bg.max():.1f} m/s")

    # 2. Initialize the Builder
    builder = DatasetBuilder(dx, dy, dz)

    # 3. Define Anomalies
    anomalies = []
    print("2. Defining anomalies...")

    # -- Anomaly A: High velocity Salt Body --
    # Center: x=1500, y=1500, z=1000
    anom_salt = EllipsoidAnomaly(
        type="salt",
        strength=0.25,        # +25% velocity perturbation
        edge_width_m=50.0,    # 50m soft boundary for smooth transition
        center=(1500, 1500, 1000),
        axes=(600, 400, 300), # Semicorner a, b, c
        R=np.eye(3)           # No rotation
    )
    anomalies.append(anom_salt)
    print(f"   Added Anomaly 1: Salt body (+25% Vp)")

    # -- Anomaly B: Low velocity Gas Pocket --
    # Center: x=800, y=2000, z=500
    # Rotate 30 degrees around Z axis
    theta = np.radians(30)
    c, s = np.cos(theta), np.sin(theta)
    R_z = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])
    
    anom_gas = EllipsoidAnomaly(
        type="gas",
        strength=-0.15,       # -15% velocity perturbation
        edge_width_m=20.0,
        center=(800, 2000, 500),
        axes=(250, 150, 100),
        R=R_z
    )
    anomalies.append(anom_gas)
    print(f"   Added Anomaly 2: Gas pocket (-15% Vp)")

    # -- Anomaly C: Salt Dome --
    print(f"   Generating random Salt Dome params...")
    # Using helper to create random params fitting our grid
    # Grid is 300x300x200 with 10m spacing => 3000x3000x2000 m
    # Note: creating params manually or via helper
    salt_params = SaltDomeAnomaly.create_random_params(
        grid_shape=(nx, ny, nz),
        grid_spacing=(dx, dy, dz),
        seed=12345
    )
    # Force some params to ensure it's visible in the middle
    salt_params.z_top_m = 400.0
    salt_params.z_base_m = 1800.0
    salt_params.stem_radius_m = 150.0
    salt_params.mid_radius_m = 300.0
    salt_params.canopy_amp_m = 400.0
    salt_params.cx_base_m = 1500.0 # Same X as anomaly 1, might overlap
    salt_params.cy_base_m = 800.0  # Different Y
    
    anom_dome = SaltDomeAnomaly(
        type="salt_dome",
        strength=0.0, # Not used by salt dome apply_to_vp (uses absolute vp), but needed for Anomaly init
        edge_width_m=salt_params.edge_width_m,
        params=salt_params
    )
    anomalies.append(anom_dome)
    print(f"   Added Anomaly 3: Salt Dome (Generated)")

    # -- Anomaly D: Igneous Dyke (Steep) --
    # Strike 45 deg, Dip 80 deg
    anom_dyke = IgneousIntrusion(
        type="dyke",
        strength=0.0, # Using absolute mix in class
        edge_width_m=20.0,
        center=(2200, 2200, 800),
        strike=45.0,
        dip=80.0,
        length_m=1200.0,
        width_m=800.0,
        thickness_m=40.0,
        roughness_amp_m=8.0 # Rocky surface
    )
    anomalies.append(anom_dyke)
    print(f"   Added Anomaly 4: Igneous Dyke (High Vp/Mag)")

    # -- Anomaly E: Igneous Sill (Flat) --
    # Strike 0, Dip 5 deg
    anom_sill = IgneousIntrusion(
        type="sill",
        strength=0.0,
        edge_width_m=30.0,
        center=(800, 800, 1500), # Deeper
        strike=0.0,
        dip=5.0,
        length_m=1000.0,
        width_m=1000.0,
        thickness_m=60.0,
        roughness_amp_m=5.0
    )
    anomalies.append(anom_sill)
    print(f"   Added Anomaly 5: Igneous Sill (High Vp/Mag)")

    # -- Anomaly F: Horizon-controlled Sill (Sill from SEGY layer) --
    if use_segy:
        print(f"   Extracting horizon for Label {target_layer_label}...")
        horizon_surf = extract_horizon_from_label(label_vol, target_label=target_layer_label, dz=dz, mode='top')
        
        # Define limits - maybe we only want a sill in part of the model
        center_x = nx * dx / 2.0
        center_y = ny * dy / 2.0
        # Estimate depth roughly
        avg_depth = np.nanmean(horizon_surf)
        
        anom_h_sill = IgneousIntrusion(
            type="sill_horizon",
            strength=0.0,
            center=(center_x, center_y, avg_depth), 
            length_m=float(nx*dx), 
            width_m=float(ny*dy),
            thickness_m=60.0,
            horizon_depths=horizon_surf, # Pass the extracted surface
            roughness_amp_m=5.0
        )
        anomalies.append(anom_h_sill)
        print(f"   Added Anomaly 6: Horizon-controlled Sill (From SEGY Label {target_layer_label})")
    
    elif not use_segy:
        # Mocking a horizon surface for demo purposes (usually read from segy)
        # Creating a sinusoidal surface
        x_idx = np.arange(nx) * dx
        y_idx = np.arange(ny) * dy
        XX, YY = np.meshgrid(x_idx, y_idx, indexing='ij') # (300, 300)
        
        # A wobbly surface at ~1200m depth
        horizon_surf = 1200.0 + 50.0 * np.sin(XX / 300.0) + 30.0 * np.cos(YY / 200.0)
        
        anom_h_sill = IgneousIntrusion(
            type="sill_horizon",
            strength=0.0,
            center=(1500, 1500, 1200), # Center is mainly for lateral fade
            length_m=2000.0, # Limit extent
            width_m=2000.0,
            thickness_m=80.0,
            horizon_depths=horizon_surf, # Pass the 2D surface array
            roughness_amp_m=2.0
        )
        anomalies.append(anom_h_sill)
        print(f"   Added Anomaly 6: Horizon-controlled Sill (Mocked Surface)")
    
    # 4. Inject Anomalies
    print("3. Injecting anomalies into model...")
    vp_final, label, X, Y, Z = builder.inject_anomalies(vp_bg, anomalies)

    # 5. Verify Results
    print("4. Result Summary:")
    print(f"   Original Vp mean: {np.mean(vp_bg):.2f}")
    print(f"   Final Vp mean:    {np.mean(vp_final):.2f}")
    
    unique_labels = np.unique(label)
    print(f"   Found labels: {unique_labels}")
    
    for lbl in unique_labels:
        if lbl == 0:
            count = np.sum(label == 0)
            print(f"     Label 0 (Background): {count} voxels ({count/label.size*100:.2f}%)")
        else:
            count = np.sum(label == lbl)
            anom_type = anomalies[lbl-1].type
            print(f"     Label {lbl} ({anom_type}):       {count} voxels ({count/label.size*100:.2f}%)")
            
    
    print("5. Visualizing...")
    nodes = cigvis.create_slices(vp_final)
    
    # 1. Salt Body (Label 1) - Cyan
    # print("   Creating surface for Salt Body...")
    # salt_mask = (label == 1).astype(np.float32)
    # if np.any(salt_mask):
    #     nodes += cigvis.create_bodys(salt_mask, level=0.5, color='cyan')

    # # 2. Gas Pocket (Label 2) - Red
    # print("   Creating surface for Gas Pocket...")
    # gas_mask = (label == 2).astype(np.float32)
    # # 3. Salt Dome (Label 3) - Yellow
    # print("   Creating surface for Salt Dome...")
    dome_mask = (label == 3).astype(np.float32)
    if np.any(dome_mask):
        nodes += cigvis.create_bodys(dome_mask, level=0.5, color='yellow')

    # 4. Dyke (Label 4) - Red (Magmatic)
    dyke_mask = (label == 4).astype(np.float32)
    if np.any(dyke_mask):
        nodes += cigvis.create_bodys(dyke_mask, level=0.5, color='red')
        
    # 5. Sill (Label 5) - Orange
    sill_mask = (label == 5).astype(np.float32)
    if np.any(sill_mask):
        nodes += cigvis.create_bodys(sill_mask, level=0.5, color='orange')
        
    # 6. Horizon Sill (Label 6) - Purple
    hsill_mask = (label == 6).astype(np.float32)
    if np.any(hsill_mask):
        nodes += cigvis.create_bodys(hsill_mask, level=0.5, color='purple')

    cigvis.plot3D(nodes, xyz_axis=True)
    # Optional: Save for checking
    # save_path = "demo_result_vp_slice.npy"
    # # Save the middle slice of Vp for quick visualization
    # mid_slice = vp_final[nz//2, :, :]
    # np.save(save_path, mid_slice)
    # print(f"\n   Saved middle depth slice to {save_path} for inspection.")
    print("=== Done ===")

if __name__ == "__main__":
    main()
