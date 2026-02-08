import numpy as np
import os
import segyio
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def convert_trends_to_labels_segy(npz_path, sgy_path, contour_num=12):
    # print(f"Processing {npz_path}...")
    try:
        data = np.load(npz_path)
        
        # Check if 'gtime' exists
        if 'gtime' not in data:
            print(f"Skipping {npz_path}: 'gtime' key not found.")
            return

        Trends = data['gtime'] # Dimensions: x * y * z
        
        # Calculate levels directly from the 3D volume statistic (Logic from visulize_layer.py)
        # Using np.linspace to divide the range of values in Trends into contour_num levels
        # We generate contour_num + 2 points and exclude the min and max to get internal levels
        if Trends.min() == Trends.max():
             # Handle flat case to avoid errors or produce single class
             levels_3d = np.array([Trends.min()])
        else:
             levels_3d = np.linspace(Trends.min(), Trends.max(), contour_num + 2)[1:-1]
        
        # Generate the label volume using the same levels
        classified_volume = np.digitize(Trends, levels_3d)
        
        # Convert to float32 for SEGY storage (SEGY format 5 is float32)
        trace_data = classified_volume.astype(np.float32)

        # Save to SEGY
        if sgy_path:
            sgy_dir = os.path.dirname(sgy_path)
            if not os.path.exists(sgy_dir):
                os.makedirs(sgy_dir)
            
            nx, ny, nz = trace_data.shape
            
            spec = segyio.spec()
            spec.ilines = np.arange(nx) + 1
            spec.xlines = np.arange(ny) + 1
            spec.samples = np.arange(nz) # Samples index, or time/depth if known
            spec.format = 5 # IEEE float
            
            with segyio.create(sgy_path, spec) as f:
                # Optimize writing: convert to float32 once and reshape
                trace_flat = trace_data.reshape(nx * ny, nz)
                f.trace[:] = trace_flat

                # Prepare header values
                ilines = np.repeat(np.arange(nx) + 1, ny)
                xlines = np.tile(np.arange(ny) + 1, nx)
                
                # Fill headers
                for tr_idx in range(nx * ny):
                    f.header[tr_idx] = {
                        segyio.TraceField.INLINE_3D: int(ilines[tr_idx]),
                        segyio.TraceField.CROSSLINE_3D: int(xlines[tr_idx]),
                        segyio.TraceField.TRACE_SAMPLE_COUNT: nz,
                        segyio.TraceField.TRACE_SAMPLE_INTERVAL: 1000
                    }
        
    except Exception as e:
        print(f"Error converting {npz_path}: {e}")

if __name__ == "__main__":
    # Configuration
    TARGET_FOLDER = '/home/wangyh/DATAFOLDER/samples/tests-river'
    # Output folder for classification labels
    TARGET_SGY_FOLDER = '/home/wangyh/DATAFOLDER/3DSeismic/AYL/3DExample/Layer_river'
    CONTOUR_NUM = 12
    TEST_ONE_FILE = False 
    
    # Collect all npz files first
    tasks = []
    for root, dirs, files in os.walk(TARGET_FOLDER):
        for file in files:
            if file.endswith('.npz'):
                npz_full_path = os.path.join(root, file)
                
                # Compute relative path to maintain structure
                rel_path = os.path.relpath(npz_full_path, TARGET_FOLDER)
                
                # Define output path
                sgy_rel_path = rel_path.replace('.npz', '.sgy')
                sgy_full_path = os.path.join(TARGET_SGY_FOLDER, sgy_rel_path)
                
                tasks.append((npz_full_path, sgy_full_path, CONTOUR_NUM))

    if TEST_ONE_FILE and tasks:
        print(f"!!! TEST MODE ENABLED !!! Only processing 1 file: {tasks[0][0]}")
        tasks = tasks[:1]

    print(f"Found {len(tasks)} files to process. Target Folder: {TARGET_SGY_FOLDER}")
    
    # Use ProcessPoolExecutor for parallel processing
    # Adjust max_workers if necessary, default is number of processors
    with ProcessPoolExecutor() as executor:
        # Submit all tasks
        futures = [executor.submit(convert_trends_to_labels_segy, src, dst, c_num) for src, dst, c_num in tasks]
        
        # Monitor progress
        for _ in tqdm(as_completed(futures), total=len(tasks), desc="Converting Trends to Labels SEGY"):
            pass
