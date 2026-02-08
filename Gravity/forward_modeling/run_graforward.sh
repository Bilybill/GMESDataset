#!/bin/bash
set -e
source /home/wangyh/anaconda3/etc/profile.d/conda.sh
conda activate torch
python main.py --config config_density_1.yaml
OUT_PATH=$(python -c "import yaml; print(yaml.safe_load(open('config_density_1.yaml'))['output']['save_path'])")
SAVE_DIR=$(dirname "$OUT_PATH")
python visualize_gravity.py --file_path "$OUT_PATH" --save_path "${SAVE_DIR}/gravity_density_1_h0.png"