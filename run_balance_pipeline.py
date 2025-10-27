import json
import logging
import os
import pickle
import copy
import argparse
from typing import List, Set

from main import run_single_experiment
from balance.workload_stream_generator import generate_workload_chunk_stream, read_queries
from balance.chunk_segmenter import segment_workloads, workload_fits_chunk
from index_selection_evaluation.selection.workload import Workload, Query
from balance.schema import Schema
from balance.workload_generator import WorkloadGenerator
from balance.query_hasher import query_to_hash

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_dict_to_workload(workload_dict, workload_generator, unify_similar_ops=False):
    queries = []
    for _, (text, freq) in enumerate(workload_dict.items()):
        _, tpl = query_to_hash(text, unify_similar_ops=unify_similar_ops, dbname=workload_generator.database_name)
        tid = workload_generator.tpl2tid[tpl]
        query = Query(tid, text, freq)
        query.columns = []
        workload_generator._store_indexable_columns(query)
        queries.append(query)
    return Workload(queries)


def main():
    logging.info("##### Starting BALANCE Pipeline: Segmentation and Policy Transfer #####")
    parser = argparse.ArgumentParser()
    parser.add_argument('--bm', type=str, default='tpchc')
    parser.add_argument('--mode', type=str, default='batch')
    parser.add_argument('--ws_file', type=str, default='')
    parser.add_argument('--wk_size', type=int, default=14)
    parser.add_argument('--ts', type=int, default=0)
    parser.add_argument('--newf', action='store_true', default=False)
    parser.add_argument('--uniComp', action='store_true', default=False)
    parser.add_argument('--random_seed', type=int, default=0, help='Set a random seed for reproducibility')
    parser.add_argument('--uni_freq', action='store_true', default=True)
    # parser.add_argument('--weight_path', type=str, default='query_files/tpch12/weight1.pkl')
    parser.add_argument('--weight_list_path', type=str, default='')
    parser.add_argument('--shuffle', action='store_true', default=False)
    parser.add_argument('--debug_print', action='store_true', help='Enable debug print statements')
    parser.add_argument('--disable_precedent_masking', action='store_const', const=True, default=None, help='Disable precedent masking')
    parser.add_argument('--enable-precedent-masking', action='store_const', const=False, default=None, help='Enable precedent masking')
    args = parser.parse_args()

    benchmark = args.bm

    # --- Step 1: Generate a continuous stream of workloads ---
    logging.info("--- Step 1: Generating a continuous workload stream ---")

    # __import__('ipdb').set_trace()

    base_config_path = f'experiments/{benchmark}_idxcount_base_config.json'

    with open(base_config_path, 'r') as f:
        base_config = json.load(f)

    # Override config with command-line seed if provided
    if args.random_seed is not None:
        base_config["random_seed"] = args.random_seed

    if args.ts:
        base_config['timesteps'] = args.ts

    if args.weight_list_path:
        base_config['workload']['varying_frequencies'] = True
        uni_freq_flag = False
    else:
        uni_freq_flag = args.uni_freq

    if uni_freq_flag:
        base_config['workload']['varying_frequencies'] = False
    else:
        base_config['workload']['varying_frequencies'] = True

    if uni_freq_flag:
        freq_label = 'uniFreq'
    else:
        freq_label = 'varyFreq'

    hash2tid = {}
    tpl2tid = {}
    if args.mode == 'batch':  # -> List[Dict[str, int]]
        # Generate 4 varied sets of 300 workloads each
        total_templates = {}
        total_templates.update(read_queries(benchmark))
        if args.mode == 'batch':
            tpl_sample_size = 14
        else:
            tpl_sample_size = args.wk_size
        base_templates = {k: total_templates[k] for k in list(total_templates.keys())[:tpl_sample_size]}
        # obtain workloads (List[str]) sperated by chunks, List[List[Dict[str, int]]]
        stream_of_chunks_dicts = generate_workload_chunk_stream(total_templates, base_templates, num_chunks=4, variation_type='query')
        # Flatten the stream and prepare for segmentation
        flat_workload_stream_dicts = [wl for chunk in stream_of_chunks_dicts for wl in chunk]

        # flat_workload_stream_dicts_new = []
        # for w in flat_workload_stream_dicts:
        #     tmp = {q: 1 for q in w}
        #     flat_workload_stream_dicts_new.append(tmp)
        # flat_workload_stream_dicts = flat_workload_stream_dicts_new

        # flat_sqls = [q for w in flat_workload_stream_dicts for q in w]
        # with open('tmp.sql', 'w') as f:
        #     f.write(''.join(flat_sqls))
    else:
        ws_file = args.ws_file
        wk_size = args.wk_size
        if 'pkl' in ws_file:
            with open(ws_file, 'rb') as f:
                flat_workload_stream_dicts = pickle.load(f)
        elif 'txt' in ws_file or 'sql' in ws_file:
            with open(ws_file, 'r') as f:
                sqls = f.readlines()
            if sqls[-1] == '':
                sqls.pop()
            num_wks = len(sqls) // wk_size
            flat_workload_stream_dicts = [{k: 1 for k in sqls[i*wk_size:(i+1)*wk_size]} for i in range(num_wks)]
        else:
            raise ValueError(f"{ws_file} can not be processed")

    i = 1
    tpl_stream = []

    dbname = ''
    if benchmark in ['tpch', 'tpchc']:
        dbname = 'indexselection_tpch___1'
    elif benchmark in ['tpcds', 'tpcdsc']:
        dbname = 'indexselection_tpcds___10'
    elif benchmark in ['ceb', 'job']:
        dbname = 'indexselection_job___1'
    else:
        raise ValueError(f"{dbname} not supported")

    # __import__('ipdb').set_trace()
    for w in flat_workload_stream_dicts:
        for qstr  in w:
            q_tpl_hash, tpl = query_to_hash(qstr, unify_similar_ops=args.uniComp, dbname=dbname)
            tpl_stream.append(tpl)
            if q_tpl_hash not in hash2tid:
                tpl2tid[tpl] = i
                hash2tid[q_tpl_hash] = i
                i += 1

    # with open(f'{args.mode}.tmp', 'wb') as f:
    #     pickle.dump(flat_workload_stream_dicts, f)
    # os._exit(0)

    schema = Schema(base_config["workload"]["benchmark"], base_config["workload"]["scale_factor"], base_config["database"], base_config["column_filters"])
    parsing_workload_generator = WorkloadGenerator(base_config["workload"], spath=base_config["workload"]["path"], workload_columns=schema.columns, random_seed=args.random_seed,
                                     database_name=dbname, experiment_id="", tpl2tid=tpl2tid, dummy=True)

    # We need the template IDs (the query text) for the segmentation logic
    workload_strset_stream = [set(wl.keys()) for wl in flat_workload_stream_dicts]

    logging.info(f"Successfully generated a flat stream of {len(workload_strset_stream)} workloads.")

    difference_threshold = 10 # As per the generator's substitution rate
    # segmented_indices = segment_workloads(workload_strset_stream, difference_threshold, hash2tid, dbname=dbname)
    chunks: List[List[Workload]] = []

    chunk_ptr = 0
    source_model_pool = []
    config = copy.deepcopy(base_config)

    if config.get('constraint_type', 'storage') == 'count':
        config['fix_index_count'] = config['constraint_value']
    else:
        config['fix_index_count'] = config.get('fix_index_count', 0)

    ws_debug = []

    weight_path_list = [os.path.join(args.weight_list_path, f"weights{i}.pkl") for i in range(1, args.wk_size + 1)]

    w_ptr = 0
    for wdict in flat_workload_stream_dicts:
        w = convert_dict_to_workload(wdict, workload_generator=parsing_workload_generator, unify_similar_ops=args.uniComp)
        ws_debug.append(w)
        w.budget = 3
        if len(chunks) == 0 or (not workload_fits_chunk(chunks[-1], w, difference_threshold, hash2tid, dbname, args.uniComp)):  # train over the workload 
            chunks.append([w])
            # a. train a new model
            if source_model_pool:   # if with source model, set the model to ppo2_BALANCE
                logging.info(f"Enabling policy transfer for Chunk {chunk_ptr} from {len(source_model_pool)} source(s).")
                config['source_model_paths'] = source_model_pool
                config['rl_algorithm']['algorithm'] = 'ppo2_BALANCE'

            chunk_config_path = f"experiment_results/{args.mode}/{benchmark}_temp_config_chunk_{chunk_ptr}.json"
            with open(chunk_config_path, 'w') as f:
                json.dump(config, f, indent=4)

            res_path = run_single_experiment(chunk_config_path, test_only=False, ts=config['timesteps'],
                        uni_freq=uni_freq_flag, fix_index_count=config['fix_index_count'], newf=args.newf,
                        input_workload=w, random_seed=args.random_seed,
                        weight_path=weight_path_list[w_ptr], shuffle=args.shuffle)
            logging.info("trained a model")

            # b. Update the source model path for the next iteration
            exp_folder = f"experiment_results/ID_{config['id']}_{config['workload']['benchmark']}_ts{config['timesteps']}_{freq_label}"
            if config['fix_index_count'] > 0:
                exp_folder += f"_idxmax{config['fix_index_count']}"
            new_model_path = os.path.join(exp_folder, "final_model.zip")
            # assert res_path == exp_folder
            if os.path.exists(new_model_path):
                source_model_pool.append(new_model_path)
                logging.info(f"Added new model to pool: {new_model_path}")
            else:
                logging.error(f"Could not find trained model for Chunk {chunk_ptr} at {new_model_path}")

            chunk_ptr = len(chunks) -1
        else:  # just append and test over the current workload
            chunks[-1].append(w)
            chunk_config_path = f"experiment_results/{args.mode}/{benchmark}_temp_config_chunk_{chunk_ptr}.json"
            res_path = run_single_experiment(chunk_config_path, test_only=True, ts=config['timesteps'],
                        uni_freq=uni_freq_flag, fix_index_count=config['fix_index_count'], newf=args.newf,
                        input_workload=w, weight_path=weight_path_list[w_ptr], shuffle=args.shuffle,
                        cli_disable_precedent_masking=args.cli_disable_precedent_masking,
                        cli_enable_precedent_masking=args.cli_enable_precedent_masking,
                        disable_precedent_masking=args.disable_precedent_masking
                        )
            print(f"test the model for new workload fits in the chunk {chunk_ptr}")
        w_ptr += 1



    # # --- Step 3: Process chunks sequentially with policy transfer ---
    # for i, chunk_indices in enumerate(segmented_indices):
    #     chunk_number = i + 1
    #     logging.info(f"\n========== PROCESSING CHUNK {chunk_number} (Workloads {chunk_indices[0]} to {chunk_indices[-1]}) ==========")

    #     # a. Get the workloads for the current chunk
    #     chunk_workload_dicts = [flat_workload_stream_dicts[j] for j in chunk_indices]
    #     chunk_workloads = [convert_dict_to_workload(wl, parsing_workload_generator, unify_similar_ops=args.uniComp) for wl in chunk_workload_dicts]

    #     # b. Prepare config
    #     config = copy.deepcopy(base_config)
    #     config['id'] = f"pipeline_chunk_{chunk_number}_query_var"
    #     if config.get('constraint_type', 'storage') == 'count':
    #         config['fix_index_count'] = config['constraint_value']
    #     else:
    #         config['fix_index_count'] = config.get('fix_index_count', 0)

    #     workload_pickle_path = f"experiment_results/{args.mode}/{benchmark}_workloads_chunk_{chunk_number}.pkl"
    #     os.makedirs(os.path.dirname(workload_pickle_path), exist_ok=True)
    #     with open(workload_pickle_path, 'wb') as f:
    #         pickle.dump(chunk_workloads, f)
    #     config['load_workloads_from_file'] = workload_pickle_path

    #     # c. Enable cumulative policy transfer
    #     if source_model_pool:
    #         logging.info(f"Enabling policy transfer for Chunk {chunk_number} from {len(source_model_pool)} source(s).")
    #         config['source_model_paths'] = source_model_pool
    #         config['rl_algorithm']['algorithm'] = 'ppo2_BALANCE'

    #     # d. Save config and run training
    #     chunk_config_path = f"experiment_results/{args.mode}/{benchmark}_temp_config_chunk_{chunk_number}.json"
    #     with open(chunk_config_path, 'w') as f:
    #         json.dump(config, f, indent=4)

    #     logging.info(f"Starting training for Chunk {chunk_number}...")

    #     logging.info(f"Training for the first workload in chunk {chunk_number}...")
    #     freq_label = 'varyFreq' if config['workload']['varying_frequencies'] else 'uniFreq'
    #     res_path = run_single_experiment(chunk_config_path, test_only=False, ts=config['timesteps'], uni_freq=freq_label,\
    #                 fix_index_count=config['fix_index_count'], newf=args.newf,\
    #                 worklaod_queries=chunk_workloads[0])
    #     logging.info(f"--- Training for Chunk {chunk_number} completed successfully. ---")

    #     # g. Update the source model path for the next iteration
    #     exp_folder = f"experiment_results/ID_{config['id']}_{config['workload']['benchmark']}_ts{config['timesteps']}_{freq_label}"
    #     if config['fix_index_count'] > 0:
    #         exp_folder += f"_idxmax{config['fix_index_count']}"
    #     new_model_path = os.path.join(exp_folder, "final_model.zip")
    #     assert res_path == exp_folder
    #     if os.path.exists(new_model_path):
    #         source_model_pool.append(new_model_path)
    #         logging.info(f"Added new model to pool: {new_model_path}")
    #     else:
    #         logging.error(f"Could not find trained model for Chunk {chunk_number} at {new_model_path}")

    logging.info("##### BALANCE Pipeline Finished #####")

if __name__ == "__main__":
    main()
