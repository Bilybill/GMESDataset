#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT="/home/wangyh/Project/GMESUni/GMESDataset/DATAFOLDER/PretrainDataset"
# 留空表示 forward_bundle.npz 直接写回各自的模型目录。
FORWARD_ROOT="/home/wangyh/Project/GMESUni/GMESDataset/DATAFOLDER/PretrainDataset_forward"

GPU_INDEX=0
GPU_POLL_INTERVAL_SEC=30
GPU_IDLE_MAX_MEMORY_MB=512
GPU_IDLE_MAX_UTIL_PCT=10

DEVICE="cuda"
GRAVITY_ALGORITHM="prism_exact"
SEISMIC_PRESET="full"
SEISMIC_BATCH_SIZE=25
SEISMIC_FREQ_MIN=5
SEISMIC_FREQ_MAX=20

cd /home/wangyh/Project/GMESUni/GMESDataset

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "Error: nvidia-smi not found. Cannot monitor GPU ${GPU_INDEX} idle state."
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

  echo "GPU ${GPU_INDEX} busy: memory=${mem_used} MiB, util=${util_used}%, compute_procs=${proc_count}. Waiting ${GPU_POLL_INTERVAL_SEC}s..."
  return 1
}

echo "Waiting for GPU ${GPU_INDEX} to become idle..."
until gpu_is_idle; do
  status=$?
  if [[ "${status}" -eq 2 ]]; then
    echo "Error: failed to query GPU ${GPU_INDEX} state via nvidia-smi."
    exit 1
  fi
  sleep "${GPU_POLL_INTERVAL_SEC}"
done

echo "GPU ${GPU_INDEX} is idle. Starting forward modeling now."
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"

python run_pretraining_forward_from_models.py \
  --model-root "$MODEL_ROOT" \
  ${FORWARD_ROOT:+--forward-root "$FORWARD_ROOT"} \
  --split-dirs tests-river \
  --device "$DEVICE" \
  --gravity-algorithm "$GRAVITY_ALGORITHM" \
  --seismic-freq-min "$SEISMIC_FREQ_MIN" \
  --seismic-freq-max "$SEISMIC_FREQ_MAX" \
  --seismic-preset "$SEISMIC_PRESET" \
  --seismic-batch-size "$SEISMIC_BATCH_SIZE" \
  --resume \
  --stop-on-error
