#!/usr/bin/env bash

CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${CONFIG_DIR}/default_suite.sh"

# Experiment 1 output directory.
# Each task/model pair will be written to:
#   ${FORWARD_OUTPUT_ROOT}/${task}_${model}/
# and logs will be written to:
#   ${FORWARD_LOG_ROOT}/YYYYMMDD_HHMMSS/
FORWARD_OUTPUT_ROOT="/home/wangyh/Project/GMESUni/GMESDataset/DATAFOLDER/ExperimentRuns/forward_exp1_braided_crossed"
FORWARD_LOG_ROOT="${FORWARD_OUTPUT_ROOT}/logs"

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
  "joint_multiphysics"
)

FORWARD_MODELS=(
  "unet"
  "fno"
)
