import json
import logging
import os
import pickle
import copy
from main import run_single_experiment
from balance.workload_generator import WorkloadGenerator
from balance.workload_stream_generator import generate_workload_stream, read_tpch_queries, TOTAL_TEMPLATES
from index_selection_evaluation.selection.workload import Workload, Query
from balance.schema import Schema

def convert_dict_to_workload(workload_dict, workload_generator):
    queries = []
    for i, (text, freq) in enumerate(workload_dict.items()):
        query = Query(i, text, freq)
        query.columns = [] # Initialize as an empty list
        workload_generator._store_indexable_columns(query)
        queries.append(query)
    return Workload(queries)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("##### Starting BALANCE Pipeline with Policy Transfer Test #####")

    # 1. Generate a workload stream in memory
    logging.info("--- Step 1: Generating workload stream ---")
    TOTAL_TEMPLATES.update(read_tpch_queries())
    base_templates = {k: TOTAL_TEMPLATES[k] for k in list(TOTAL_TEMPLATES.keys())[:14]}
    stream = generate_workload_stream(base_templates, num_chunks=2, variation_type='query')
    logging.info(f"Successfully generated a stream with {len(stream)} chunks.")

    source_model_path = None
    base_config_path = 'experiments/tpchc_test_config.json'

    # 2. Process chunks sequentially
    for i, chunk_workloads in enumerate(stream):
        chunk_number = i + 1
        logging.info(f"\n========== PROCESSING CHUNK {chunk_number} ==========")

        # a. Prepare configuration for the current chunk
        with open(base_config_path, 'r') as f:
            config = json.load(f)

        # This generator is only used to provide context for column parsing
        schema = Schema(config["workload"]["benchmark"], config["workload"]["scale_factor"], config["database"], config["column_filters"])
        workload_generator = WorkloadGenerator(config["workload"], spath=config["workload"]["path"], workload_columns=schema.columns, random_seed=0, database_name="", experiment_id="")
        chunk_workloads = [convert_dict_to_workload(wl, workload_generator) for wl in chunk_workloads]

        # Create a unique ID and folder for this chunk's experiment
        config['id'] = f"pipeline_chunk_{chunk_number}"
        config['fix_index_count'] = config.get('fix_index_count', 0)

        # Save this chunk's workloads to a temporary pickle file
        workload_pickle_path = f"experiment_results/temp_workloads_chunk_{chunk_number}.pkl"
        with open(workload_pickle_path, 'wb') as f:
            pickle.dump(chunk_workloads, f)
        config['load_workloads_from_file'] = workload_pickle_path

        # b. Enable policy transfer for the second chunk
        if chunk_number > 1 and source_model_path:
            logging.info(f"Enabling policy transfer for Chunk {chunk_number}.")
            config['source_model_path'] = source_model_path
            # Use the custom PPO2 algorithm that handles transfer
            config['rl_algorithm']['name'] = 'ppo2_BALANCE'

        # c. Save the new temporary config file
        chunk_config_path = f"experiment_results/temp_config_chunk_{chunk_number}.json"
        with open(chunk_config_path, 'w') as f:
            json.dump(config, f, indent=4)

        # d. Run the training for the current chunk
        logging.info(f"Starting training for Chunk {chunk_number}...")
        try:
            run_single_experiment(chunk_config_path)
            logging.info(f"--- Training for Chunk {chunk_number} completed successfully. ---")
            # Update the source model path for the next iteration
            exp_folder = f"experiment_results/ID_{config['id']}_{config['workload']['benchmark']}_ts{config['timesteps']}_uniFreq_idxmax{config['fix_index_count']}"
            source_model_path = os.path.join(exp_folder, "final_model.zip")
            logging.info(f"Saved model for this chunk at: {source_model_path}")

        except Exception as e:
            logging.error(f"Training for Chunk {chunk_number} FAILED: {e}", exc_info=True)
            break # Stop the pipeline if a chunk fails

    logging.info("\n##### BALANCE Pipeline Finished #####")

if __name__ == "__main__":
    main()
