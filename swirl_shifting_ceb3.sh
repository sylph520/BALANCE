wk_type_ori="${1:-ceb_custom_12}"  # The new input format (using _ instead of -)

# 1. Normalize the string: Remove _custom, then replace the FIRST remaining _ with a -
# This converts: ceb_12  --> ceb-12
# And:           tpcds-3_12 --> tpcds-3_12 (no change needed here)
input_string="${wk_type_ori/_custom/}"
input_string="${input_string/_/-}" # Replace the first underscore with a hyphen

process_string() {
  local input_string="$1"
  local prefix="${input_string%%-*}" # Extract the prefix (e.g., ceb, tpcds)
  local num_part="${input_string#*-}" # Extract the number(s) part (e.g., 12, 3_12)

  # Check if the number part contains an underscore (two numbers)
  if [[ "$num_part" == *"_"* ]]; then
    # Case 1: Two numbers (e.g., 3_12)
    local first_num="${num_part%%_*}" # Extracts 3
    local second_num="${num_part#*_}" # Extracts 12
    # Output: tpcds12-3
    echo "${prefix}${second_num}-${first_num}"
  else
    # Case 2: One number (e.g., 12)
    # Output: ceb12
    echo "${prefix}${num_part}"
  fi
}

wk_type=$(process_string "$input_string")
echo "Input: ${wk_type_ori}"
echo "Normalized Input: ${input_string}"
echo "Output: ${wk_type}"

if [ -n $2 ];
then
  extra=$2
fi

if [ ! -f ${wk_type}_swirl_shifting3${extra}.log ];
then
  python -m selection conf/config_rl_ceb_idxub3_wl_embedder.json --raylog False --train False --state_mode 6 --restore_path ~/project/ise2_ray_results_test/PPO/PPO-job1_2000_128_30_8e-06_256x256-20251005_001543_sm6_rm1_rk-1_seed1234hjgppbl/checkpoint_000120/checkpoint-120 --workload_runs 1 --sql_file  /home/sclai/project/ise2/res/workload_shifting/${wk_type}/workload1.txt  >> logs/${wk_type}_swirl_shifting3${extra}.log 2>&1
  python -m selection conf/config_rl_ceb_idxub3_wl_embedder.json --raylog False --train False --state_mode 6 --restore_path ~/project/ise2_ray_results_test/PPO/PPO-job1_2000_128_30_8e-06_256x256-20251005_001543_sm6_rm1_rk-1_seed1234hjgppbl/checkpoint_000120/checkpoint-120 --workload_runs 1 --sql_file  /home/sclai/project/ise2/res/workload_shifting/${wk_type}/workload2.txt  >> logs/${wk_type}_swirl_shifting3${extra}.log 2>&1
  python -m selection conf/config_rl_ceb_idxub3_wl_embedder.json --raylog False --train False --state_mode 6 --restore_path ~/project/ise2_ray_results_test/PPO/PPO-job1_2000_128_30_8e-06_256x256-20251005_001543_sm6_rm1_rk-1_seed1234hjgppbl/checkpoint_000120/checkpoint-120 --workload_runs 1 --sql_file  /home/sclai/project/ise2/res/workload_shifting/${wk_type}/workload3.txt  >> logs/${wk_type}_swirl_shifting3${extra}.log 2>&1
  python -m selection conf/config_rl_ceb_idxub3_wl_embedder.json --raylog False --train False --state_mode 6 --restore_path ~/project/ise2_ray_results_test/PPO/PPO-job1_2000_128_30_8e-06_256x256-20251005_001543_sm6_rm1_rk-1_seed1234hjgppbl/checkpoint_000120/checkpoint-120 --workload_runs 1 --sql_file  /home/sclai/project/ise2/res/workload_shifting/${wk_type}/workload4.txt  >> logs/${wk_type}_swirl_shifting3${extra}.log 2>&1
  python -m selection conf/config_rl_ceb_idxub3_wl_embedder.json --raylog False --train False --state_mode 6 --restore_path ~/project/ise2_ray_results_test/PPO/PPO-job1_2000_128_30_8e-06_256x256-20251005_001543_sm6_rm1_rk-1_seed1234hjgppbl/checkpoint_000120/checkpoint-120 --workload_runs 1 --sql_file  /home/sclai/project/ise2/res/workload_shifting/${wk_type}/workload5.txt  >> logs/${wk_type}_swirl_shifting3${extra}.log 2>&1
  python -m selection conf/config_rl_ceb_idxub3_wl_embedder.json --raylog False --train False --state_mode 6 --restore_path ~/project/ise2_ray_results_test/PPO/PPO-job1_2000_128_30_8e-06_256x256-20251005_001543_sm6_rm1_rk-1_seed1234hjgppbl/checkpoint_000120/checkpoint-120 --workload_runs 1 --sql_file  /home/sclai/project/ise2/res/workload_shifting/${wk_type}/workload6.txt  >> logs/${wk_type}_swirl_shifting3${extra}.log 2>&1
  python -m selection conf/config_rl_ceb_idxub3_wl_embedder.json --raylog False --train False --state_mode 6 --restore_path ~/project/ise2_ray_results_test/PPO/PPO-job1_2000_128_30_8e-06_256x256-20251005_001543_sm6_rm1_rk-1_seed1234hjgppbl/checkpoint_000120/checkpoint-120 --workload_runs 1 --sql_file  /home/sclai/project/ise2/res/workload_shifting/${wk_type}/workload7.txt  >> logs/${wk_type}_swirl_shifting3${extra}.log 2>&1
  python -m selection conf/config_rl_ceb_idxub3_wl_embedder.json --raylog False --train False --state_mode 6 --restore_path ~/project/ise2_ray_results_test/PPO/PPO-job1_2000_128_30_8e-06_256x256-20251005_001543_sm6_rm1_rk-1_seed1234hjgppbl/checkpoint_000120/checkpoint-120 --workload_runs 1 --sql_file  /home/sclai/project/ise2/res/workload_shifting/${wk_type}/workload8.txt  >> logs/${wk_type}_swirl_shifting3${extra}.log 2>&1
  python -m selection conf/config_rl_ceb_idxub3_wl_embedder.json --raylog False --train False --state_mode 6 --restore_path ~/project/ise2_ray_results_test/PPO/PPO-job1_2000_128_30_8e-06_256x256-20251005_001543_sm6_rm1_rk-1_seed1234hjgppbl/checkpoint_000120/checkpoint-120 --workload_runs 1 --sql_file  /home/sclai/project/ise2/res/workload_shifting/${wk_type}/workload9.txt  >> logs/${wk_type}_swirl_shifting3${extra}.log 2>&1
  python -m selection conf/config_rl_ceb_idxub3_wl_embedder.json --raylog False --train False --state_mode 6 --restore_path ~/project/ise2_ray_results_test/PPO/PPO-job1_2000_128_30_8e-06_256x256-20251005_001543_sm6_rm1_rk-1_seed1234hjgppbl/checkpoint_000120/checkpoint-120 --workload_runs 1 --sql_file  /home/sclai/project/ise2/res/workload_shifting/${wk_type}/workload10.txt >> logs/${wk_type}_swirl_shifting3${extra}.log 2>&1
  python -m selection conf/config_rl_ceb_idxub3_wl_embedder.json --raylog False --train False --state_mode 6 --restore_path ~/project/ise2_ray_results_test/PPO/PPO-job1_2000_128_30_8e-06_256x256-20251005_001543_sm6_rm1_rk-1_seed1234hjgppbl/checkpoint_000120/checkpoint-120 --workload_runs 1 --sql_file  /home/sclai/project/ise2/res/workload_shifting/${wk_type}/workload11.txt >> logs/${wk_type}_swirl_shifting3${extra}.log 2>&1
  python -m selection conf/config_rl_ceb_idxub3_wl_embedder.json --raylog False --train False --state_mode 6 --restore_path ~/project/ise2_ray_results_test/PPO/PPO-job1_2000_128_30_8e-06_256x256-20251005_001543_sm6_rm1_rk-1_seed1234hjgppbl/checkpoint_000120/checkpoint-120 --workload_runs 1 --sql_file  /home/sclai/project/ise2/res/workload_shifting/${wk_type}/workload12.txt >> logs/${wk_type}_swirl_shifting3${extra}.log 2>&1
fi
echo "${wk_type}_swirl_shifting3${extra}.log"
awk -F': ' '/Overall Costs/ {print $2}' logs/${wk_type}_swirl_shifting3${extra}.log | paste -sd ','
