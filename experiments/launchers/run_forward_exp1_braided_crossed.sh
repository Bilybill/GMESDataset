#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${PROJECT_ROOT}/experiments/configs/forward/exp1_braided_crossed.sh"
EXP1_PHASE="${EXP1_PHASE:-phase1}"

case "${EXP1_PHASE}" in
  phase1)
    DEFAULT_TASK_FILTER="rho_to_gravity,chi_to_magnetic"
    DEFAULT_MODEL_FILTER="unet,fno"
    ;;
  phase2)
    DEFAULT_TASK_FILTER="res_to_mt"
    DEFAULT_MODEL_FILTER="unet,fno"
    ;;
  phase3)
    DEFAULT_TASK_FILTER="vp_to_seismic"
    DEFAULT_MODEL_FILTER="unet,fno"
    ;;
  phase4)
    DEFAULT_TASK_FILTER="joint_multiphysics"
    DEFAULT_MODEL_FILTER="unet,fno"
    ;;
  full)
    DEFAULT_TASK_FILTER="rho_to_gravity,chi_to_magnetic,res_to_mt,vp_to_seismic,joint_multiphysics"
    DEFAULT_MODEL_FILTER="unet,fno"
    ;;
  *)
    echo "Error: unsupported EXP1_PHASE='${EXP1_PHASE}'. Expected one of: phase1, phase2, phase3, phase4, full."
    exit 1
    ;;
esac

export TASK_FILTER="${TASK_FILTER:-${DEFAULT_TASK_FILTER}}"
export MODEL_FILTER="${MODEL_FILTER:-${DEFAULT_MODEL_FILTER}}"

echo "[exp1-forward] phase=${EXP1_PHASE}"
echo "[exp1-forward] task_filter=${TASK_FILTER}"
echo "[exp1-forward] model_filter=${MODEL_FILTER}"
echo "[exp1-forward] config=${CONFIG_PATH}"
echo "[exp1-forward] results_root=/home/wangyh/Project/GMESUni/GMESDataset/DATAFOLDER/ExperimentRuns/forward_exp1_braided_crossed"
echo "[exp1-forward] per-run outputs: results_root/<task>_<model>/"
echo "[exp1-forward] logs: results_root/logs/<timestamp>/"

exec bash "${SCRIPT_DIR}/run_forward_suite.sh" "${CONFIG_PATH}"
