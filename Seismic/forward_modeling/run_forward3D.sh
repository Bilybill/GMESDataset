#!/bin/bash

# 确保脚本发生错误时停止
set -e

# 激活 conda 环境
source /home/wangyh/anaconda3/etc/profile.d/conda.sh
conda activate torch

echo "Starting 3D Forward Modeling..."

# 检查是否存在 config_3D.yaml，如果没有则提示用户
if [ ! -f "config_3D.yaml" ]; then
    echo "Error: config_3D.yaml not found. Please create it first."
    exit 1
fi

# 运行 Python 主程序，指定 3D 配置文件
python main.py --config config_3D.yaml

# 从配置文件解析 save_path
RAW_PATH=$(python -c "import yaml; print(yaml.safe_load(open('config_3D.yaml'))['output']['save_path'])")

# 模拟 main.py 的后缀处理逻辑 (确保读取的是最终生成的 .npz 文件)
if [[ "$RAW_PATH" == *".npy" ]]; then
    SAVE_PATH="${RAW_PATH%.npy}.npz"
elif [[ "$RAW_PATH" != *".npz" ]]; then
    SAVE_PATH="${RAW_PATH}.npz"
else
    SAVE_PATH="$RAW_PATH"
fi

SAVE_DIR=$(dirname "$SAVE_PATH")
QC_DIR="${SAVE_DIR}/qc_figs"

echo "Forward modeling complete. Data saved to: $SAVE_PATH"
echo "Starting QC visualization..."
echo "QC figures will be saved to: $QC_DIR"

# 运行可视化脚本进行 QC
python visualize_npz_gathers.py --file "$SAVE_PATH" --sort_by_offset --save_dir "$QC_DIR"
./run_viz_shot.sh
