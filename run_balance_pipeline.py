import json
import logging
import os
import pickle
import copy
from main import run_single_experiment
from balance.workload_stream_generator import generate_workload_stream, read_tpch_queries, TOTAL_TEMPLATES
from balance.chunk_segmenter import segment_workloads
from index_selection_evaluation.selection.workload import Workload, Query
from balance.schema import Schema
from balance.workload_generator import WorkloadGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_dict_to_workload(workload_dict, workload_generator):
    queries = []
    for i, (text, freq) in enumerate(workload_dict.items()):
        query = Query(i, text, freq)
        query.columns = []
        workload_generator._store_indexable_columns(query)
        queries.append(query)
    return Workload(queries)

def main():
    logging.info("##### Starting BALANCE Pipeline: Segmentation and Policy Transfer #####")

    # --- Step 1: Generate a continuous stream of workloads ---
    logging.info("--- Step 1: Generating a continuous workload stream ---")
    TOTAL_TEMPLATES.update(read_tpch_queries())
    base_templates = {k: TOTAL_TEMPLATES[k] for k in list(TOTAL_TEMPLATES.keys())[:14]}

    # Generate 4 varied sets of 300 workloads each
    stream_of_chunks_dicts = generate_workload_stream(base_templates, num_chunks=4, variation_type='query')

    # Flatten the stream and prepare for segmentation
    flat_workload_stream_dicts = [wl for chunk in stream_of_chunks_dicts for wl in chunk]
    # We need the template IDs (the query text) for the segmentation logic
    workload_stream_for_segmentation = [set(wl.keys()) for wl in flat_workload_stream_dicts]
    logging.info(f"Successfully generated a flat stream of {len(workload_stream_for_segmentation)} workloads.")

    # --- Step 2: Segment the stream into chunks ---
    difference_threshold = 10 # As per the generator's substitution rate
    logging.info(f"--- Step 2: Segmenting stream with a {difference_threshold}% threshold ---")
    segmented_indices = segment_workloads(workload_stream_for_segmentation, difference_threshold)

    logging.info(f"========== Segmentation Complete: Identified {len(segmented_indices)} Chunks ==========")

    # --- Step 3: Process chunks sequentially with policy transfer ---
    source_model_pool = []
    base_config_path = 'experiments/tpchc_test_config.json'

    with open(base_config_path, 'r') as f:
        base_config = json.load(f)
    schema = Schema(base_config["workload"]["benchmark"], base_config["workload"]["scale_factor"], base_config["database"], base_config["column_filters"])
    parsing_workload_generator = WorkloadGenerator(base_config["workload"], spath=base_config["workload"]["path"], workload_columns=schema.columns, random_seed=0, database_name="", experiment_id="")

    for i, chunk_indices in enumerate(segmented_indices):
        chunk_number = i + 1
        logging.info(f"\n========== PROCESSING CHUNK {chunk_number} (Workloads {chunk_indices[0]} to {chunk_indices[-1]}) ==========")

        # a. Get the workloads for the current chunk
        chunk_workload_dicts = [flat_workload_stream_dicts[j] for j in chunk_indices]
        chunk_workloads = [convert_dict_to_workload(wl, parsing_workload_generator) for wl in chunk_workload_dicts]

        # b. Prepare config
        config = copy.deepcopy(base_config)
        config['id'] = f"pipeline_chunk_{chunk_number}_query_var"
        config['fix_index_count'] = config.get('fix_index_count', 0)

        workload_pickle_path = f"experiment_results/temp_workloads_chunk_{chunk_number}.pkl"
        with open(workload_pickle_path, 'wb') as f:
            pickle.dump(chunk_workloads, f)
        config['load_workloads_from_file'] = workload_pickle_path

        # c. Enable cumulative policy transfer
        if source_model_pool:
            logging.info(f"Enabling policy transfer for Chunk {chunk_number} from {len(source_model_pool)} source(s).")
            config['source_model_paths'] = source_model_pool
            config['rl_algorithm']['name'] = 'ppo2_BALANCE'

        # d. Save config and run training
        chunk_config_path = f"experiment_results/temp_config_chunk_{chunk_number}.json"
        with open(chunk_config_path, 'w') as f:
            json.dump(config, f, indent=4)

        logging.info(f"Starting training for Chunk {chunk_number}...")
        try:
            run_single_experiment(chunk_config_path)
            logging.info(f"--- Training for Chunk {chunk_number} completed successfully. ---")

            # g. Update the source model path for the next iteration
            freq_label = 'varyFreq' if config['workload']['varying_frequencies'] else 'uniFreq'
            exp_folder = f"experiment_results/ID_{config['id']}_{config['workload']['benchmark']}_ts{config['timesteps']}_{freq_label}"
            if config['fix_index_count'] > 0:
                exp_folder += f"_idxmax{config['fix_index_count']}"
            new_model_path = os.path.join(exp_folder, "final_model.zip")
            if os.path.exists(new_model_path):
                source_model_pool.append(new_model_path)
                logging.info(f"Added new model to pool: {new_model_path}")
            else:
                logging.error(f"Could not find trained model for Chunk {chunk_number} at {new_model_path}")

        except Exception as e:
            logging.error(f"Training for Chunk {chunk_number} FAILED: {e}", exc_info=True)
            break

    logging.info("##### BALANCE Pipeline Finished #####")

if __name__ == "__main__":
    main()
