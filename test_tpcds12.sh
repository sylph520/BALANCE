#!/bin/bash

# --- Default Configuration ---
# DEFAULT_MODEL_DIR="tblog_ID_Test_Experiment_1_Index_Count_TPCDSC_ts100000_dmxsz50_PPO2-cliprange0p3-learning_rate5em05-n_steps512-nminibatches4-noptepochs30-gamma0p5_uniFreq_idxmax5OFFprecedentMasking_8"
# DEFAULT_TS="100000"
DEFAULT_TS="6000"
# DEFAULT_MODEL_DIR=""
MAX_ITERATIONS=12
MODEL_DIR_ARG=""

# --- Argument Parsing ---
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --imax)
      MAX_ITERATIONS="$2"
      shift # past argument
      shift # past value
      ;;
    --ts)
      MAX_ITERATIONS="$3"
      shift # past argument
      shift # past value
      ;;
    *)
      # The first non-flag argument is treated as the model directory
      if [ -z "$MODEL_DIR_ARG" ]; then
        MODEL_DIR_ARG="$1"
      fi
      shift # past argument
      ;;
  esac
done

# --- Set Variables ---
# Use provided model directory or fall back to the default
if [ -n "$MODEL_DIR_ARG" ]; then
  MODEL_DIR="$MODEL_DIR_ARG"
else
  MODEL_DIR="$DEFAULT_MODEL_DIR"
fi

# Construct the full model path
MODEL_PATH="tensor_log/${MODEL_DIR}/final_model.zip"

# Fixed parameters (with updated QIDs)
CONFIG_FILE="experiments/tpcdsc_conf/tpcdsc_index_count_tune.json"
# TEST_WORKLOAD_QIDS="16, 18, 4, 12, 14, 6, 10, 11, 19, 9, 8, 20, 5, 1, 13, 7, 15, 3, 2, 17"
FIX_INDEX_COUNT=5

# --- Main Execution Logic ---
echo "Starting test sequence for model: ${MODEL_PATH}"
echo "Running up to ${MAX_ITERATIONS} workloads."
echo "--------------------------------------------------"

# Loop through workloads from 1 to MAX_ITERATIONS
for i in $(seq 1 $MAX_ITERATIONS); do
  WORKLOAD_FILE="query_files/tpcds12/workload${i}.txt"
  WEIGHT_FILE="query_files/tpcds12/weights${i}.pkl"

  echo "Running test for workload ${i} (${WORKLOAD_FILE}) with weights (${WEIGHT_FILE})..."
  python main.py \
    --config "${CONFIG_FILE}" \
    # --load_model "${MODEL_PATH}" \
    --test_only \
    --fix_index_count "${FIX_INDEX_COUNT}" \
	--ts "${DEFAULT_TS}" \
    --input_workload_path "${WORKLOAD_FILE}" \
    --weight_path "${WEIGHT_FILE}" \
  	--test_model_freq uniFreq

  echo "--------------------------------------------------"
done

echo "Test sequence completed."
