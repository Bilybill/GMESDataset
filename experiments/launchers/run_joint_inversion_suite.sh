#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${1:-${PROJECT_ROOT}/experiments/configs/inversion/default_suite.sh}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
# shellcheck disable=SC1090
source "${CONFIG_PATH}"

activate_conda_env_if_needed
cd "${PROJECT_ROOT}"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
SUITE_LOG_DIR="${INVERSION_LOG_ROOT}/${TIMESTAMP}"
ensure_directory "${SUITE_LOG_DIR}"

echo "[joint-inversion] project_root=${PROJECT_ROOT}"
echo "[joint-inversion] config=${CONFIG_PATH}"
echo "[joint-inversion] device=${INVERSION_DEVICE}"
echo "[joint-inversion] logs=${SUITE_LOG_DIR}"

run_name="joint_inversion_$(IFS=_; echo "${INVERSION_MODALITIES[*]}")"
log_path="${SUITE_LOG_DIR}/${run_name}.log"

train_cmd=(
  python
  experiments/train_joint_inversion.py
  --root "${INVERSION_ROOT}"
  --include-top-levels "${INCLUDE_TOP_LEVELS[@]}"
  --modalities "${INVERSION_MODALITIES[@]}"
  --device "${INVERSION_DEVICE}"
  --batch-size "${INVERSION_BATCH_SIZE}"
  --epochs "${INVERSION_EPOCHS}"
  --lr "${INVERSION_LR}"
  --weight-decay "${INVERSION_WEIGHT_DECAY}"
  --num-workers "${INVERSION_NUM_WORKERS}"
  --val-fraction "${INVERSION_VAL_FRACTION}"
  --seed "${INVERSION_SEED}"
  --embedding-dim "${INVERSION_EMBEDDING_DIM}"
  --fusion-hidden-dim "${INVERSION_FUSION_HIDDEN_DIM}"
  --decoder-base-channels "${INVERSION_DECODER_BASE_CHANNELS}"
  --target-shape "${INVERSION_TARGET_SHAPE[@]}"
  --max-train-samples "${INVERSION_MAX_TRAIN_SAMPLES}"
  --max-val-samples "${INVERSION_MAX_VAL_SAMPLES}"
  --max-heldout-samples "${INVERSION_MAX_HELDOUT_SAMPLES}"
  --output-root "${INVERSION_OUTPUT_ROOT}"
)

if [[ "${#INVERSION_DEVELOPMENT_SOURCE_PREFIXES[@]}" -gt 0 ]]; then
  train_cmd+=(--development-source-prefixes "${INVERSION_DEVELOPMENT_SOURCE_PREFIXES[@]}")
fi
if [[ "${#INVERSION_HELDOUT_SOURCE_PREFIXES[@]}" -gt 0 ]]; then
  train_cmd+=(--heldout-source-prefixes "${INVERSION_HELDOUT_SOURCE_PREFIXES[@]}")
fi
if [[ "${INVERSION_RESUME:-0}" == "1" ]]; then
  train_cmd+=(--resume)
fi

run_logged_command "${log_path}" "${train_cmd[@]}"
