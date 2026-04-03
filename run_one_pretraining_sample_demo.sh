#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VELOCITY_ROOT="/home/wangyh/DATAFOLDER/3DSeismic/AYLModel/ALLvelocity"
SAMPLE_ROOT="${SCRIPT_DIR}/DATAFOLDER/samples"
OUTPUT_ROOT="${SCRIPT_DIR}/DATAFOLDER/Cache/PretrainDatasetOneModelDemo"

ANOMALY_TYPE="igneous_swarm"
SPLIT_DIR="tests-river"
MAX_SAMPLES=1

DEVICE="cuda"
SEISMIC_PRESET="light"
SEISMIC_BATCH_SIZE=9
SHOT_INDEX=0

python "${SCRIPT_DIR}/build_pretraining_dataset.py" \
  --stage full \
  --velocity-root "${VELOCITY_ROOT}" \
  --sample-root "${SAMPLE_ROOT}" \
  --label-source-mode samples \
  --label-contour-num 12 \
  --output-root "${OUTPUT_ROOT}" \
  --split-dirs "${SPLIT_DIR}" \
  --max-samples "${MAX_SAMPLES}" \
  --anomaly-types "${ANOMALY_TYPE}" \
  --device "${DEVICE}" \
  --gravity-algorithm prism_exact \
  --anomaly_mode background \
  --seismic-preset "${SEISMIC_PRESET}" \
  --seismic-batch-size "${SEISMIC_BATCH_SIZE}" \
  --save-previews

SAMPLE_DIR="$(find "${OUTPUT_ROOT}" -type f -path "*/${ANOMALY_TYPE}/model_bundle.npz" | sort | head -n 1 | xargs dirname)"

if [[ -z "${SAMPLE_DIR}" ]]; then
  echo "No generated sample directory found under ${OUTPUT_ROOT}" >&2
  exit 1
fi

python "${SCRIPT_DIR}/visualize_pretraining_sample.py" \
  --sample-dir "${SAMPLE_DIR}" \
  --shot-index "${SHOT_INDEX}"

echo "Done."
echo "Sample dir: ${SAMPLE_DIR}"
echo "Visualization dir: ${SAMPLE_DIR}/viz_sample"
