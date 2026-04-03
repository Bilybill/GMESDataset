#!/usr/bin/env bash
set -euo pipefail

VELOCITY_ROOT="/home/wangyh/DATAFOLDER/3DSeismic/AYLModel/ALLvelocity"
SAMPLE_ROOT="/home/wangyh/Project/GMESUni/GMESDataset/DATAFOLDER/samples"
OUTPUT_ROOT="/home/wangyh/Project/GMESUni/GMESDataset/DATAFOLDER/preview_outputs/preview_10_models"
ANOMALY_RANDOM_CONFIG="/home/wangyh/Project/GMESUni/GMESDataset/configs/pretraining_anomaly_randomization.yaml"

MAX_SAMPLES=10
SEED_OFFSET=0

cd /home/wangyh/Project/GMESUni/GMESDataset

mkdir -p "$OUTPUT_ROOT"

python build_pretraining_dataset.py \
  --stage models \
  --velocity-root "$VELOCITY_ROOT" \
  --sample-root "$SAMPLE_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --split-dirs tests-river \
  --anomaly-types igneous_swarm igneous_stock gas hydrate brine_fault massive_sulfide salt_dome sediment_basement serpentinized \
  --anomaly-selection-mode random_one \
  --variants-per-model 1 \
  --seed-offset "$SEED_OFFSET" \
  --anomaly-random-config "$ANOMALY_RANDOM_CONFIG" \
  --max-samples "$MAX_SAMPLES" \
  --resume \
  --stop-on-error

find "$OUTPUT_ROOT" -name model_bundle.npz -print0 | sort -z | while IFS= read -r -d '' model_bundle; do
  sample_dir="$(dirname "$model_bundle")"
  echo "Visualizing: $sample_dir"
  python visualize_pretraining_sample.py --sample-dir "$sample_dir"
done

echo "Preview complete: $OUTPUT_ROOT"
