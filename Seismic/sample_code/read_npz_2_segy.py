#%%
import numpy as np
import os
from matplotlib import pyplot as plt
import segyio
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def convert_npz_to_segy(npz_path, sgy_path, bin_path=None, trends_range=(2000, 6000), details_range=(-500, 1000), mdetails_range=(-200, 200)):
    # print(f"Processing {npz_path}...")
    try:
        data = np.load(npz_path)
        # print(list(data.keys())) 
        
        Details = data['formatS']
        MDetails = data['formatD']
        Trends = data['gtime'] # Dimensions: x * y * z
        
        trends_vmin, trends_vmax = trends_range
        Trends = (Trends - Trends.min()) / (Trends.max() - Trends.min())
        Trends = trends_vmin + Trends * (trends_vmax - trends_vmin)

        details_vmin, details_vmax = details_range
        Details = (Details - Details.min()) / (Details.max() - Details.min())
        Details = details_vmin + Details * (details_vmax - details_vmin)
        
        mdetails_vmin, mdetails_vmax = mdetails_range
        MDetails = (MDetails - MDetails.min()) / (MDetails.max() - MDetails.min())
        MDetails = mdetails_vmin + MDetails * (mdetails_vmax - mdetails_vmin)

        velocity = Trends + Details + MDetails
        
        # Save to BIN
        if bin_path:
            bin_dir = os.path.dirname(bin_path)
            if not os.path.exists(bin_dir):
                os.makedirs(bin_dir)
            velocity.astype(np.float32).tofile(bin_path)

        # Save to SEGY
        if sgy_path:
            sgy_dir = os.path.dirname(sgy_path)
            if not os.path.exists(sgy_dir):
                os.makedirs(sgy_dir)
            
            nx, ny, nz = velocity.shape
            
            spec = segyio.spec()
            spec.ilines = np.arange(nx) + 1
            spec.xlines = np.arange(ny) + 1
            spec.samples = np.arange(nz) # Samples index, or time/depth if known
            spec.format = 5 # IEEE float
            
            with segyio.create(sgy_path, spec) as f:
                # Optimize writing: convert to float32 once and reshape
                velocity_flat = velocity.reshape(nx * ny, nz).astype(np.float32)
                f.trace[:] = velocity_flat

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
        
        # print(f"Saved: {sgy_path}")
        
    except Exception as e:
        print(f"Error converting {npz_path}: {e}")

if __name__ == "__main__":
    # Load data from .npz file
    TARGET_FOLDER = '/home/wangyh/DATAFOLDER/samples/tests-choas'
    TARGET_SGY_FOLDER = '/home/wangyh/DATAFOLDER/3DSeismic/AYL/3DExample/Velocity_choas'
    TEST_ONE_FILE = False # 修改这里为 False 以转换所有文件
    EXPORT_SEGY = True
    EXPORT_BIN = False
    
    # Collect all npz files first
    tasks = []
    for root, dirs, files in os.walk(TARGET_FOLDER):
        for file in files:
            if file.endswith('.npz'):
                npz_full_path = os.path.join(root, file)
                
                # Compute relative path to maintain structure
                rel_path = os.path.relpath(npz_full_path, TARGET_FOLDER)
                
                sgy_full_path = None
                if EXPORT_SEGY:
                    sgy_rel_path = rel_path.replace('.npz', '.sgy')
                    sgy_full_path = os.path.join(TARGET_SGY_FOLDER, sgy_rel_path)
                
                bin_full_path = None
                if EXPORT_BIN:
                    bin_rel_path = rel_path.replace('.npz', '.bin')
                    bin_full_path = os.path.join(TARGET_SGY_FOLDER, bin_rel_path)
                
                tasks.append((npz_full_path, sgy_full_path, bin_full_path))

    if TEST_ONE_FILE and tasks:
        print(f"!!! TEST MODE ENABLED !!! Only processing 1 file: {tasks[0][0]}")
        tasks = tasks[:1]

    # Use ProcessPoolExecutor for parallel processing
    # Adjust max_workers if necessary, default is number of processors
    with ProcessPoolExecutor() as executor:
        # Submit all tasks
        futures = [executor.submit(convert_npz_to_segy, src, dst_sgy, dst_bin) for src, dst_sgy, dst_bin in tasks]
        
        # Monitor progress
        for _ in tqdm(as_completed(futures), total=len(tasks), desc="Converting files"):
            pass