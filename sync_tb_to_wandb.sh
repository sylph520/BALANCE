#!/usr/bin/env bash
set -euo pipefail

TB_ROOT="${1:-tensor_log/gridsearch_tblog}"
PROJECT="${2:-balance}"
ENTITY="${3:-your_wandb_entity}"

if [[ ! -d "$TB_ROOT" ]]; then
  echo "TensorBoard root not found: $TB_ROOT" >&2
  exit 1
fi

find "$TB_ROOT" -mindepth 1 -maxdepth 1 -type d | sort | while read -r run_dir; do
  echo "Flattening and syncing $run_dir"
  python tensor_log/bulk_log_arg.py \
    --proj_name "$PROJECT" \
    --entity "$ENTITY" \
    --run_dir "$run_dir"
done
