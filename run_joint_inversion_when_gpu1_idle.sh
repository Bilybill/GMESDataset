#!/usr/bin/env bash
set -euo pipefail

# Wait for GPU 1 to become idle, then launch the GMES-3D joint inversion
# baseline in the torch conda environment.
#
# Default task:
#   gravity + magnetic + MT + seismic -> vp + rho + res + chi
#
# Outputs:
#   ${INVERSION_OUTPUT_ROOT}/joint_inversion_gravity_magnetic_mt_seismic_to_vp_rho_res_chi/
#
# Live log:
#   ${INVERSION_OUTPUT_ROOT}/wait_gpu1_joint_inversion_<timestamp>.log

PROJECT_ROOT="/home/wangyh/Project/GMESUni/GMESDataset"
CONDA_SH="${CONDA_SH:-/home/wangyh/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-torch}"

GPU_INDEX="${GPU_INDEX:-1}"
GPU_POLL_INTERVAL_SEC="${GPU_POLL_INTERVAL_SEC:-30}"
GPU_IDLE_MAX_MEMORY_MB="${GPU_IDLE_MAX_MEMORY_MB:-1024}"
GPU_IDLE_MAX_UTIL_PCT="${GPU_IDLE_MAX_UTIL_PCT:-10}"

INVERSION_ROOT="${INVERSION_ROOT:-${PROJECT_ROOT}/DATAFOLDER/PretrainDataset_forward}"
INVERSION_OUTPUT_ROOT="${INVERSION_OUTPUT_ROOT:-${PROJECT_ROOT}/DATAFOLDER/ExperimentRuns/joint_inversion}"
INVERSION_LOG_ROOT="${INVERSION_LOG_ROOT:-${INVERSION_OUTPUT_ROOT}/logs}"

INVERSION_DEVICE="${INVERSION_DEVICE:-cuda}"
INVERSION_BATCH_SIZE="${INVERSION_BATCH_SIZE:-1}"
INVERSION_EPOCHS="${INVERSION_EPOCHS:-40}"
INVERSION_LR="${INVERSION_LR:-5e-4}"
INVERSION_WEIGHT_DECAY="${INVERSION_WEIGHT_DECAY:-1e-4}"
INVERSION_NUM_WORKERS="${INVERSION_NUM_WORKERS:-2}"
INVERSION_SEED="${INVERSION_SEED:-42}"
INVERSION_TARGET_SHAPE="${INVERSION_TARGET_SHAPE:-64 64 64}"
INVERSION_RESUME="${INVERSION_RESUME:-0}"

cd "${PROJECT_ROOT}"
mkdir -p "${INVERSION_OUTPUT_ROOT}" "${INVERSION_LOG_ROOT}"

RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
WAIT_LOG="${INVERSION_OUTPUT_ROOT}/wait_gpu${GPU_INDEX}_joint_inversion_${RUN_TIMESTAMP}.log"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${WAIT_LOG}"
}

if ! command -v nvidia-smi >/dev/null 2>&1; then
  log "Error: nvidia-smi not found. Cannot monitor GPU ${GPU_INDEX}."
  exit 1
fi

if [[ ! -f "${CONDA_SH}" ]]; then
  log "Error: conda activation script not found: ${CONDA_SH}"
  exit 1
fi

gpu_is_idle() {
  local gpu_line mem_used util_used proc_output proc_count
  gpu_line="$(nvidia-smi --id="${GPU_INDEX}" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)"
  if [[ -z "${gpu_line}" ]]; then
    return 2
  fi

  mem_used="$(echo "${gpu_line}" | cut -d',' -f1 | tr -d ' ')"
  util_used="$(echo "${gpu_line}" | cut -d',' -f2 | tr -d ' ')"

  proc_output="$(nvidia-smi --id="${GPU_INDEX}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
  proc_output="$(echo "${proc_output}" | sed '/^$/d;/No running processes found/d')"
  proc_count="$(echo "${proc_output}" | sed '/^$/d' | wc -l | tr -d ' ')"

  if [[ "${proc_count}" == "0" ]] && [[ "${mem_used}" -le "${GPU_IDLE_MAX_MEMORY_MB}" ]] && [[ "${util_used}" -le "${GPU_IDLE_MAX_UTIL_PCT}" ]]; then
    return 0
  fi

  log "GPU ${GPU_INDEX} busy: memory=${mem_used} MiB, util=${util_used}%, compute_procs=${proc_count}. Waiting ${GPU_POLL_INTERVAL_SEC}s..."
  return 1
}

log "Waiting for GPU ${GPU_INDEX} to become idle..."
until gpu_is_idle; do
  status=$?
  if [[ "${status}" -eq 2 ]]; then
    log "Error: failed to query GPU ${GPU_INDEX} state via nvidia-smi."
    exit 1
  fi
  sleep "${GPU_POLL_INTERVAL_SEC}"
done

log "GPU ${GPU_INDEX} is idle. Starting joint inversion."

# Map physical GPU 1 to visible cuda:0 inside the training process.  The
# launcher still receives INVERSION_DEVICE=cuda by default, which resolves to
# the only visible GPU.
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export CONDA_ENV
export INVERSION_ROOT
export INVERSION_OUTPUT_ROOT
export INVERSION_LOG_ROOT
export INVERSION_DEVICE
export INVERSION_BATCH_SIZE
export INVERSION_EPOCHS
export INVERSION_LR
export INVERSION_WEIGHT_DECAY
export INVERSION_NUM_WORKERS
export INVERSION_SEED
export INVERSION_TARGET_SHAPE
export INVERSION_RESUME

set +u
# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
set -u

log "Activated conda env: ${CONDA_DEFAULT_ENV:-unknown}"
log "Launching experiments/launchers/run_joint_inversion_suite.sh"

bash experiments/launchers/run_joint_inversion_suite.sh 2>&1 | tee -a "${WAIT_LOG}"
