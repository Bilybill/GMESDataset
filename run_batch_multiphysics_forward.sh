#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
SEISMIC_PRESET="${SEISMIC_PRESET:-light}"
SEISMIC_BATCH_SIZE="${SEISMIC_BATCH_SIZE:-0}"
SHOT_INDEX="${SHOT_INDEX:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/DATAFOLDER/Cache/ForwardBatch}"
VP_SEGY="${VP_SEGY:-/home/wangyh/DATAFOLDER/3DSeismic/AYLModel/3DExample/Velocity_choas/braided/AYL-00000.sgy}"
LABEL_SEGY="${LABEL_SEGY:-/home/wangyh/DATAFOLDER/3DSeismic/AYLModel/3DExample/Layer_choas/braided/AYL-00000.sgy}"
RUN_APP="${RUN_APP:-0}"
SKIP_MT="${SKIP_MT:-0}"
SKIP_SEISMIC="${SKIP_SEISMIC:-0}"
MT_FREQ_MIN="${MT_FREQ_MIN:-}"
MT_FREQ_MAX="${MT_FREQ_MAX:-}"

DEFAULT_ANOMALIES=(
  igneous_swarm
  massive_sulfide
  brine_fault
  salt_dome
  serpentinized
)

print_help() {
  cat <<EOF
Usage:
  bash ${0##*/} [anomaly_type ...]

Behavior:
  - Without positional arguments, runs a default batch:
      ${DEFAULT_ANOMALIES[*]}
  - With positional arguments, only runs the specified anomaly types.

Environment overrides:
  PYTHON_BIN           Python executable. Default: ${PYTHON_BIN}
  DEVICE               auto | cpu | cuda. Default: ${DEVICE}
  SEISMIC_PRESET       full | light. Default: ${SEISMIC_PRESET}
  SEISMIC_BATCH_SIZE   Shots per seismic batch. Default: ${SEISMIC_BATCH_SIZE}
  SHOT_INDEX           Shot index used by plot_saved_forward_data.py. Default: ${SHOT_INDEX}
  OUTPUT_ROOT          Root folder for per-anomaly outputs. Default: ${OUTPUT_ROOT}
  VP_SEGY              Input velocity SEGY. Default: ${VP_SEGY}
  LABEL_SEGY           Input label SEGY. Default: ${LABEL_SEGY}
  RUN_APP              Set to 1 to pass --run_app to plotting script. Default: ${RUN_APP}
  SKIP_MT              Set to 1 to pass --skip_mt. Default: ${SKIP_MT}
  SKIP_SEISMIC         Set to 1 to pass --skip_seismic. Default: ${SKIP_SEISMIC}
  MT_FREQ_MIN          Optional MT min frequency in Hz. Must be paired with MT_FREQ_MAX.
  MT_FREQ_MAX          Optional MT max frequency in Hz. Must be paired with MT_FREQ_MIN.

Examples:
  bash ${0##*/}
  DEVICE=cpu SEISMIC_PRESET=light bash ${0##*/} igneous_swarm massive_sulfide
  MT_FREQ_MIN=0.1 MT_FREQ_MAX=1000 bash ${0##*/} serpentinized
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  print_help
  exit 0
fi

if [[ $# -gt 0 ]]; then
  ANOMALIES=("$@")
else
  ANOMALIES=("${DEFAULT_ANOMALIES[@]}")
fi

if [[ -n "${MT_FREQ_MIN}" || -n "${MT_FREQ_MAX}" ]]; then
  if [[ -z "${MT_FREQ_MIN}" || -z "${MT_FREQ_MAX}" ]]; then
    echo "[ERROR] MT_FREQ_MIN and MT_FREQ_MAX must be set together." >&2
    exit 1
  fi
fi

mkdir -p "${OUTPUT_ROOT}"

echo "============================================================"
echo "GMESDataset batch forward modeling"
echo "Python:            ${PYTHON_BIN}"
echo "Device:            ${DEVICE}"
echo "Seismic preset:    ${SEISMIC_PRESET}"
echo "Seismic batch:     ${SEISMIC_BATCH_SIZE}"
echo "Plot shot index:   ${SHOT_INDEX}"
echo "Output root:       ${OUTPUT_ROOT}"
echo "Anomalies:         ${ANOMALIES[*]}"
echo "============================================================"

for anomaly in "${ANOMALIES[@]}"; do
  RUN_DIR="${OUTPUT_ROOT}/${anomaly}"
  PLOT_DIR="${RUN_DIR}/custom_plots"
  mkdir -p "${RUN_DIR}" "${PLOT_DIR}"

  echo
  echo "============================================================"
  echo "[RUN] anomaly_type=${anomaly}"
  echo "  save_dir:   ${RUN_DIR}"
  echo "  plot_dir:   ${PLOT_DIR}"
  echo "============================================================"

  FORWARD_CMD=(
    "${PYTHON_BIN}"
    "${SCRIPT_DIR}/run_multiphysics_forward.py"
    "--save_dir" "${RUN_DIR}"
    "--vp_segy" "${VP_SEGY}"
    "--label_segy" "${LABEL_SEGY}"
    "--anomaly-type" "${anomaly}"
    "--device" "${DEVICE}"
    "--seismic-preset" "${SEISMIC_PRESET}"
    "--seismic-batch-size" "${SEISMIC_BATCH_SIZE}"
  )

  if [[ "${SKIP_MT}" == "1" ]]; then
    FORWARD_CMD+=("--skip_mt")
  fi
  if [[ "${SKIP_SEISMIC}" == "1" ]]; then
    FORWARD_CMD+=("--skip_seismic")
  fi
  if [[ -n "${MT_FREQ_MIN}" && -n "${MT_FREQ_MAX}" ]]; then
    FORWARD_CMD+=("--mt-freq-min" "${MT_FREQ_MIN}" "--mt-freq-max" "${MT_FREQ_MAX}")
  fi

  echo "[CMD] ${FORWARD_CMD[*]}"
  "${FORWARD_CMD[@]}"

  PLOT_CMD=(
    "${PYTHON_BIN}"
    "${SCRIPT_DIR}/plot_saved_forward_data.py"
    "--save_dir" "${RUN_DIR}"
    "--output_dir" "${PLOT_DIR}"
    "--shot_index" "${SHOT_INDEX}"
  )

  if [[ "${RUN_APP}" == "1" ]]; then
    PLOT_CMD+=("--run_app")
  fi

  echo "[CMD] ${PLOT_CMD[*]}"
  "${PLOT_CMD[@]}"

  echo "[DONE] ${anomaly}"
done

echo
echo "All anomaly runs completed."
echo "Results are under: ${OUTPUT_ROOT}"
