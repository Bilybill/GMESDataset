#!/usr/bin/env bash
set -euo pipefail

VELOCITY_ROOT="/home/wangyh/DATAFOLDER/3DSeismic/AYLModel/ALLvelocity"
SAMPLE_ROOT="/home/wangyh/Project/GMESUni/GMESDataset/DATAFOLDER/samples"
OUTPUT_ROOT="/home/wangyh/Project/GMESUni/GMESDataset/DATAFOLDER/PretrainDataset"
ANOMALY_RANDOM_CONFIG="/home/wangyh/Project/GMESUni/GMESDataset/configs/pretraining_anomaly_randomization.yaml"

VARIANTS_PER_MODEL=1
SEED_OFFSET=0

cd /home/wangyh/Project/GMESUni/GMESDataset

python build_pretraining_dataset.py \
  --stage models \
  --velocity-root "$VELOCITY_ROOT" \
  --sample-root "$SAMPLE_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --split-dirs train-river \
  --anomaly-types igneous_swarm igneous_stock gas hydrate brine_fault massive_sulfide salt_dome sediment_basement serpentinized \
  --anomaly-selection-mode random_one \
  --variants-per-model "$VARIANTS_PER_MODEL" \
  --seed-offset "$SEED_OFFSET" \
  --anomaly-random-config "$ANOMALY_RANDOM_CONFIG" \
  --resume \
  --stop-on-error
