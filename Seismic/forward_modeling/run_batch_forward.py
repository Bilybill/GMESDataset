import os
import yaml
import subprocess
import copy
import sys
from tqdm import tqdm
from loguru import logger

# --- Configuration ---
SOURCE_ROOT = "/home/wangyh/DATAFOLDER/3DSeismic/AYLModel/ALLvelocity"
OUTPUT_ROOT = "/home/wangyh/DATAFOLDER/3DSeismic/AYLModel/ALLVelocity3DSeismic"
TARGET_SUBDIRS = ["tests-choas", "tests-river"]
BASE_CONFIG_PATH = "config_3D.yaml"
PYTHON_EXEC = sys.executable
MAIN_SCRIPT = "main.py"

# Configure logger to save to "batch_run.log" in current directory
logger.add("batch_run.log", format="{time} {level} {message}", rotation="10 MB")

def run_batch():
    logger.info("Initializing batch run...")
    # 1. Check Files
    if not os.path.exists(BASE_CONFIG_PATH):
        print(f"Error: Base config file '{BASE_CONFIG_PATH}' not found in current directory.")
        return
    if not os.path.exists(MAIN_SCRIPT):
        print(f"Error: Main script '{MAIN_SCRIPT}' not found in current directory.")
        return

    # 2. Load and Prepare Base Config
    print(f"Loading base config from {BASE_CONFIG_PATH}...")
    with open(BASE_CONFIG_PATH, 'r') as f:
        base_config = yaml.safe_load(f)

    # Update global settings for batch run as requested
    base_config['simulation']['measure_time'] = False
    if 'acquisition' not in base_config: base_config['acquisition'] = {}
    base_config['acquisition']['show_geometry'] = False

    # 3. Scan for Binary Files
    tasks = []
    print(f"Scanning for .bin files in {SOURCE_ROOT}...")
    
    for subdir in TARGET_SUBDIRS:
        src_dir = os.path.join(SOURCE_ROOT, subdir)
        if not os.path.exists(src_dir):
            print(f"Warning: Source directory {src_dir} does not exist. Skipping.")
            continue
            
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                if file.endswith(".bin"):
                    src_file_path = os.path.join(root, file)
                    
                    # Calculate relative path to mirror structure
                    # e.g. tests-choas/subfolder/file.bin
                    rel_path = os.path.relpath(src_file_path, SOURCE_ROOT)
                    
                    # Construct output path (change extension to .npz)
                    out_file_rel = os.path.splitext(rel_path)[0] + ".npz"
                    out_file_path = os.path.join(OUTPUT_ROOT, out_file_rel)
                    
                    tasks.append({
                        'src': src_file_path,
                        'dst': out_file_path
                    })

    if not tasks:
        print("No .bin files found in the specified directories.")
        return

    print(f"Found {len(tasks)} tasks.")
    print("-" * 50)

    # 4. Execute Tasks
    # We create a temporary config file for each task to avoid concurrency issues if we were parallel,
    # but here we run sequentially to respect GPU resources.
    
    temp_config_filename = "temp_batch_config.yaml"

    try:
        for i, task in enumerate(tqdm(tasks, desc="Processing")):
            src = task['src']
            dst = task['dst']
            
            # Skip if output already exists? (Optional, maybe not needed unless requested)
            # if os.path.exists(dst):
            #     continue
            
            # Prepare specific config
            current_config = copy.deepcopy(base_config)
            current_config['model']['file_path'] = src
            current_config['output']['save_path'] = dst
            
            # Save temp config
            with open(temp_config_filename, 'w') as f:
                yaml.dump(current_config, f)
            
            # Run main.py
            # We redirect stdout/stderr to avoid cluttering the progress bar, 
            # or we can log to a file.
            cmd = [PYTHON_EXEC, MAIN_SCRIPT, "--config", temp_config_filename]
            
            # Capture output to check for errors
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                stderr_tail = '\n'.join(result.stderr.splitlines()[-20:])
                logger.error(f"FAILED processing file: {src}")
                logger.error(f"Return Code: {result.returncode}")
                logger.error(f"Stderr Tail:\n{stderr_tail}")
                
                tqdm.write(f"\n[Error] Failed processing {src}. See batch_run.log for details.")
            else:
                logger.info(f"SUCCESS processing file: {src}")
    
    except KeyboardInterrupt:
        print("\nBatch processing interrupted by user.")
    finally:
        # Cleanup
        if os.path.exists(temp_config_filename):
            os.remove(temp_config_filename)
        print("\nBatch processing finished.")

if __name__ == "__main__":
    run_batch()
