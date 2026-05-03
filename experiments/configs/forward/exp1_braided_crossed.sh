#!/usr/bin/env bash

CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${CONFIG_DIR}/default_suite.sh"

# Experiment 1 output directory.
# Each task/model pair will be written to:
#   ${FORWARD_OUTPUT_ROOT}/${task}_${model}/
# and logs will be written to:
#   ${FORWARD_LOG_ROOT}/YYYYMMDD_HHMMSS/
FORWARD_OUTPUT_ROOT="${FORWARD_OUTPUT_ROOT:-/home/wangyh/Project/GMESUni/GMESDataset/DATAFOLDER/ExperimentRuns/forward_exp1_braided_crossed}"
FORWARD_LOG_ROOT="${FORWARD_LOG_ROOT:-${FORWARD_OUTPUT_ROOT}/logs}"

# Use only braided + crossed from train-river as the development pool.
# With val_fraction=0.15, this gives:
#   train = 680, val = 120, held-out = 120
FORWARD_VAL_FRACTION="0.15"

INCLUDE_TOP_LEVELS=(
  "train-river"
  "tests-river"
)

FORWARD_DEVELOPMENT_SOURCE_PREFIXES=(
  "train-river/braided"
  "train-river/crossed"
)

FORWARD_HELDOUT_SOURCE_PREFIXES=()

FORWARD_TASKS=(
  "rho_to_gravity"
  "chi_to_magnetic"
  "res_to_mt"
  "vp_to_seismic"
  "vp_source_to_seismic_shot"
  "joint_multiphysics"
)

FORWARD_MODELS=(
  "unet"
  "fno"
  "shot_film"
)

declare -A FORWARD_TASK_MODELS=(
  ["rho_to_gravity"]="unet fno"
  ["chi_to_magnetic"]="unet fno"
  ["res_to_mt"]="unet fno"
  ["vp_to_seismic"]="unet fno"
  ["vp_source_to_seismic_shot"]="shot_film"
  ["joint_multiphysics"]="unet fno"
)

# Phase-3 dedicated overrides for the shot-conditioned seismic benchmark.
# These env vars affect only vp_source_to_seismic_shot when this config is used.
FORWARD_BATCH_SIZE["vp_source_to_seismic_shot"]="${SEISMIC_SHOT_BATCH_SIZE:-3}"
FORWARD_EPOCHS["vp_source_to_seismic_shot"]="${SEISMIC_SHOT_EPOCHS:-50}"
FORWARD_LR["vp_source_to_seismic_shot"]="${SEISMIC_SHOT_LR:-5e-4}"

declare -A FORWARD_TASK_WEIGHT_DECAY=(
  ["vp_source_to_seismic_shot"]="${SEISMIC_SHOT_WEIGHT_DECAY:-1e-4}"
)

declare -A FORWARD_TASK_NUM_WORKERS=(
  ["vp_source_to_seismic_shot"]="${SEISMIC_SHOT_NUM_WORKERS:-8}"
)

FORWARD_SEED="${SEISMIC_SHOT_SEED:-42}"
