#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_batch.sh --bm <benchmark> [--config path] [--ts timesteps] [--imax N] [--model_path /abs/path/to/model.zip] [--model_id run_folder]
  --bm          Benchmark name (e.g. tpcds12, tpch12). Required.
  --config      Path to config JSON (default: experiments/tpcdsc_conf/tpcdsc_index_count_tune.json).
  --ts          Timesteps used when resolving experiment folders (default: 3000).
  --imax        Limit to the first N workloads (default: all).
  --model_path  Full path to a model zip.
  --model_id    TensorBoard run folder; expands to tensor_log/<model_id>/final_model.zip.
                Ignored when --model_path is given.
EOF
  exit 1
}

BM=""
MODEL_PATH=""
MODEL_ID=""
TS=3000
IMAX=0
CONFIG_PATH="experiments/tpcdsc_conf/tpcdsc_index_count_tune.json"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bm)         [[ $# -gt 1 ]] || usage; BM="$2"; shift 2;;
    --config)     [[ $# -gt 1 ]] || usage; CONFIG_PATH="$2"; shift 2;;
    --ts)         [[ $# -gt 1 ]] || usage; TS="$2"; shift 2;;
    --imax)       [[ $# -gt 1 ]] || usage; IMAX="$2"; shift 2;;
    --model_path) [[ $# -gt 1 ]] || usage; MODEL_PATH="$2"; shift 2;;
    --model_id)   [[ $# -gt 1 ]] || usage; MODEL_ID="$2"; shift 2;;
    -h|--help)    usage;;
    *) EXTRA_ARGS+=("$1"); shift;;
  esac
done

[[ -n "$BM" ]] || usage

BASE_DIR="query_files/${BM}"
if [[ ! -d "$BASE_DIR" ]]; then
  echo "Benchmark directory not found: $BASE_DIR" >&2
  exit 1
fi

MODEL_ARG=()
if [[ -n "$MODEL_PATH" ]]; then
  MODEL_ARG=(--load_model "$MODEL_PATH")
elif [[ -n "$MODEL_ID" ]]; then
  MODEL_ARG=(--load_model "tensor_log/${MODEL_ID}/final_model.zip")
fi

shopt -s nullglob
workload_glob=("$BASE_DIR"/workload*.txt)
shopt -u nullglob
if [[ ${#workload_glob[@]} -eq 0 ]]; then
  echo "No workload*.txt files found in $BASE_DIR" >&2
  exit 1
fi
IFS=$'\n' read -r -d '' -a workloads < <(printf '%s\n' "${workload_glob[@]}" | sort -V && printf '\0')

if [[ "$IMAX" =~ ^[0-9]+$ && "$IMAX" -gt 0 && ${#workloads[@]} -gt "$IMAX" ]]; then
  workloads=("${workloads[@]:0:IMAX}")
fi

for workload_path in "${workloads[@]}"; do
  fname=$(basename "$workload_path")
  idx=${fname//[!0-9]/}
  [[ -n "$idx" ]] || continue

  weight_path="$BASE_DIR/weights${idx}.pkl"
  if [[ ! -f "$weight_path" ]]; then
    echo "Skipping ${fname}: missing weights${idx}.pkl" >&2
    continue
  fi

  echo ">>> Running ${BM} workload ${idx}"
  CMD=(python main.py
    --config "$CONFIG_PATH"
    --test_only
    --fix_index_count 5
    --ts "$TS"
    --input_workload_path "$workload_path"
    --weight_path "$weight_path"
    --test_model_freq=uniFreq
    "${MODEL_ARG[@]}"
    "${EXTRA_ARGS[@]}"
  )
  printf 'CMD:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  "${CMD[@]}"
done
