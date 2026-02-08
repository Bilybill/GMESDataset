#!/bin/bash

# 激活 conda 环境
source /home/wangyh/anaconda3/etc/profile.d/conda.sh
conda activate torch

# 默认配置文件
CONFIG_FILE="config_3D.yaml"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: $CONFIG_FILE not found."
    exit 1
fi

# 从配置文件解析 save_path
RAW_PATH=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['output']['save_path'])")

# 处理后缀 logic (同 main.py)
if [[ "$RAW_PATH" == *".npy" ]]; then
    DATA_FILE="${RAW_PATH%.npy}.npz"
elif [[ "$RAW_PATH" != *".npz" ]]; then
    DATA_FILE="${RAW_PATH}.npz"
else
    DATA_FILE="$RAW_PATH"
fi

echo "Target Data File: $DATA_FILE"

# 检查数据文件是否存在
if [ ! -f "$DATA_FILE" ]; then
    echo "Error: Data file $DATA_FILE not found. Run forward modeling first."
    exit 1
fi

# 运行可视化脚本，传递所有额外参数
# python visualize_shot.py --file "$DATA_FILE" --output /home/wangyh/DATAFOLDER/3DSeismic/Cache/3Doutputexp "$@"
python visualize_shot_GPT.py --file "$DATA_FILE" --output /home/wangyh/DATAFOLDER/3DSeismic/Cache/3Doutputexp --mode inline "$@"

python visualize_shot_GPT.py --file "$DATA_FILE" --output /home/wangyh/DATAFOLDER/3DSeismic/Cache/3Doutputexp --mode offset_binned "$@"

python visualize_shot_GPT.py --file "$DATA_FILE" --output /home/wangyh/DATAFOLDER/3DSeismic/Cache/3Doutputexp --mode grid_slice "$@"
