import sys
import os
import numpy as np
import segyio

# Ensure cigvis is in path
sys.path.append('/home/wangyh/Project/cigvis')
import cigvis

def read_segy(path):
    print(f"Reading SEGY file: {path}")
    with segyio.open(path, ignore_geometry=True) as f:
        f.mmap()
        n_traces = f.tracecount
        nt = f.samples.size
        
        # Infer dimensions nx, ny assuming full volume
        # We know from generation that ilines vary slower than xlines
        # ilines = np.repeat(np.arange(nx) + 1, ny)
        # xlines = np.tile(np.arange(ny) + 1, nx)
        try:
            il = f.attributes(segyio.TraceField.INLINE_3D)[:]
            xl = f.attributes(segyio.TraceField.CROSSLINE_3D)[:]
            
            nx = len(np.unique(il))
            ny = len(np.unique(xl))
        except:
             nx = 0
             ny = 0
        
        if nx * ny != n_traces:
             print(f"Warning: Calculated dimensions {nx}x{ny}={nx*ny} do not match trace count {n_traces}. Trying to infer from geometry...")
             # Fallback or just assume square if nx=ny
             if int(np.sqrt(n_traces))**2 == n_traces:
                 nx = ny = int(np.sqrt(n_traces))
             else:
                 # Try 256
                 if n_traces == 256*256:
                     nx = 256
                     ny = 256
                 else:
                     raise ValueError(f"Cannot determine dimensions from headers cleanly. Traces: {n_traces}")

        print(f"Dimensions inferred: {nx} x {ny} x {nt}")
        
        data = np.stack([t for t in f.trace])
        data = data.reshape(nx, ny, nt)
        return data

def main():
    # Define paths
    # Using 'braided/AYL-00000.sgy' as a sample
    base_dir = '/home/wangyh/DATAFOLDER/3DSeismic/AYL/3DExample'
    vel_dir = os.path.join(base_dir, 'Velocity_river/braided')
    lbl_dir = os.path.join(base_dir, 'Layer_river/braided')
    
    filename = 'AYL-00000.sgy'
    vel_path = os.path.join(vel_dir, filename)
    lbl_path = os.path.join(lbl_dir, filename)
    
    if not os.path.exists(vel_path):
        print(f"Velocity file not found: {vel_path}")
        # Try finding any file
        for root, dirs, files in os.walk(base_dir):
            for f in files:
                if f.endswith('.sgy') and 'Velocity' in root:
                    print(f"Found alternative: {os.path.join(root, f)}")
        return
    if not os.path.exists(lbl_path):
        print(f"Label file not found: {lbl_path}")
        return

    # Read data
    velocity = read_segy(vel_path)
    labels = read_segy(lbl_path)

    # Validate shapes
    if velocity.shape != labels.shape:
        print(f"Shape mismatch: Velocity {velocity.shape} vs Labels {labels.shape}")
        return
    
    # Normalize data for visualization
    # Velocity usually doesn't need much normalization for 'gray' cmap
    # Labels should be integers, let's ensure
    labels = labels.astype(np.float32)
    
    print("Preparing visualization...")
    
    # 1. Create slices for Velocity (Background)
    # Automatically pick some slices position
    nx, ny, nz = velocity.shape
    # explicit: pos = [[nx//2], [ny//2], [nz//2]] for inline, crossline, time slices
    
    nodes = cigvis.create_slices(velocity, cmap='gray', pos=[[nx//2], [ny//2], [nz//2]])
    
    # 2. Overlay Labels
    # Use a discrete colormap or a colorful one with transparency
    # 'tab20' is good for categorical data.
    # alpha=0.6 to see the velocity underneath
    fg_cmap = cigvis.colormap.set_alpha('tab20', alpha=0.6)
    
    # Add mask overlay
    # Cigvis add_mask doc says: nodes = cigvis.add_mask(nodes, volume, cmaps=...)
    # It overlays on the existing slices in 'nodes'
    nodes = cigvis.add_mask(nodes, labels, cmaps=fg_cmap)
    
    # 3. Add colorbars
    # Velocity colorbar
    # nodes += cigvis.create_colorbar_from_nodes(nodes, 'Velocity', select='slices')
    
    # Label colorbar
    # This might show the full gradient if not careful, but tab20 is discrete-ish.
    nodes += cigvis.create_colorbar_from_nodes(nodes, 'Labels', select='mask')

    # 4. Plot
    print("Displaying plot window...")
    save_path = '3D_visualization_cigvis.png'
    cigvis.plot3D(nodes, size=(1000, 800), savename=save_path, xyz_axis=True)
    print(f"Plot saved to {save_path} (if headless, otherwise shown in window)")

if __name__ == '__main__':
    main()

