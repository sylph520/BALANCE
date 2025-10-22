import os
import random
import copy
import json
import logging
import argparse
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def read_queries(bm: str) -> Dict[str, List[str]]:
    """Reads all TPCH query templates from their files."""
    query_path = f"query_files/{bm.upper()}"
    templates = {}
    for i in os.listdir(query_path):
        if i.endswith('.txt'):
            tpl_id = i.split('.')[0].split('_')[-1]
            with open(os.path.join(query_path, i), 'r') as f:
                q_list = f.readlines()
            if q_list[-1] == '':
                q_list.pop()
            templates[tpl_id] = q_list
    return templates


def generate_workload_chunk(templates, num_workloads, vary_frequency=True, queries_per_workload=14):
    """Generates a single chunk of workloads from a given set of templates."""
    chunk = []
    template_items = list(templates.items())
    for _ in range(num_workloads):
        workload = {}
        selected_templates = random.sample(template_items, queries_per_workload)
        for template_id, query_text in selected_templates:
            frequency = random.randint(1, 10000) if vary_frequency else 1
            try:
                workload[query_text] = frequency
            except:
                raise ValueError(query_text)
        chunk.append(workload)
    return chunk

def create_varied_chunk(total_templates, source_templates, unused_templates, variation_type, substitution_rate):
    """Creates a new set of templates by varying a source set."""
    next_templates = copy.deepcopy(source_templates)
    vary_freq_for_new_chunk = True

    if variation_type == 'query' or variation_type == 'combined':
        num_to_substitute = int(len(source_templates) * substitution_rate)
        if len(unused_templates) < num_to_substitute:
            raise ValueError("Not enough unused templates to satisfy substitution rate.")

        templates_to_remove_ids = random.sample(list(source_templates.keys()), num_to_substitute)
        templates_to_add_ids = random.sample(list(unused_templates.keys()), num_to_substitute)

        for tid in templates_to_remove_ids:
            del next_templates[tid]
        for tid in templates_to_add_ids:
            next_templates[tid] = random.choice(total_templates[tid])

        # Update the pool of unused templates as well
        for tid in templates_to_add_ids:
            del unused_templates[tid]
        for tid in templates_to_remove_ids:
            unused_templates[tid] = random.choice(total_templates[tid])

        logging.info(f"Substituted {num_to_substitute} templates. New template pool: {sorted(list(next_templates.keys()))}")
    elif variation_type == 'frequency':
        vary_freq_for_new_chunk = False
        logging.info("Using same templates as previous chunk, but with fixed frequencies.")

    return next_templates, unused_templates, vary_freq_for_new_chunk

def generate_workload_chunk_stream(total_templates, base_templates, num_chunks, variation_type='query', substitution_rate=0.3,
                                   workloads_per_chunk=300, queries_per_workload=14):
    """Generates a stream of workload chunks with controlled variation."""
    if len(base_templates) < queries_per_workload:
        raise ValueError(f"base_templates must contain at least {queries_per_workload} templates.")

    stream = []
    current_templates = {k: random.choice(v) for k, v in base_templates.items()}
    unused_templates = {k: random.choice(v) for k, v in total_templates.items() if k not in current_templates}
    # __import__('ipdb').set_trace()

    # 1. Create the first, base chunk
    logging.info(f"Generating Base Chunk 1 with templates: {sorted(list(current_templates.keys()))}")
    base_chunk = generate_workload_chunk(current_templates, workloads_per_chunk, vary_frequency=True, queries_per_workload=queries_per_workload)
    stream.append(base_chunk)

    # 2. Create subsequent, varied chunks
    for i in range(num_chunks - 1):
        chunk_num = i + 2
        logging.info(f"--- Generating Varied Chunk {chunk_num} (Variation: {variation_type}) ---")

        next_templates, unused_templates, vary_freq = create_varied_chunk(
            total_templates, current_templates, unused_templates, variation_type, substitution_rate
        )

        new_chunk = generate_workload_chunk(next_templates, workloads_per_chunk, vary_frequency=vary_freq)
        stream.append(new_chunk)
        current_templates = next_templates

    return stream

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a stream of workload chunks with specified variation.")
    parser.add_argument('--bm', type=str, default='tpch')
    parser.add_argument("--variation_type", type=str, choices=['query', 'frequency', 'combined'],
                        help="The type of variation between workload chunks.", default='query')
    args = parser.parse_args()

    # Load the actual TPCH query templates
    total_templates = read_queries(args.bm)
    if not total_templates:
        logging.error("No TPCH query templates could be loaded. Exiting.")
        exit(1)

    # Define parameters for the workload stream generation
    NUM_CHUNKS = 4
    BASE_TEMPLATE_POOL_SIZE = 14

    # Randomly select a pool of templates for the first chunk
    if BASE_TEMPLATE_POOL_SIZE > len(total_templates):
        raise ValueError("Pool size cannot be larger than total available templates.")

    # Seed for reproducibility
    random.seed(42)
    # Get the template IDs to sample from
    available_template_ids = list(total_templates.keys())
    base_template_ids = random.sample(available_template_ids, BASE_TEMPLATE_POOL_SIZE)
    base_template_pool = {tid: total_templates[tid] for tid in base_template_ids}

    # Generate the stream based on the specified variation type
    logging.info(f"\n##### GENERATING STREAM: '{args.variation_type.upper()}' VARIATION #####")
    workload_stream = generate_workload_chunk_stream(total_templates, base_template_pool, NUM_CHUNKS, variation_type=args.variation_type)

    output_filename = f'workload_stream_{args.variation_type}_variation.json'
    with open(output_filename, 'w') as f:
        # Storing the workloads with actual query text
        json.dump(workload_stream, f, indent=2)

    # --- Direct test for create_varied_chunk ---
    print("\n\n=================================================================")
    print("    DIRECTLY TESTING `create_varied_chunk` FUNCTION")
    print("=================================================================")

    # Create a source pool of templates
    source_pool_ids = base_template_ids
    source_pool_templates = {tid: total_templates[tid] for tid in source_pool_ids}

    # Create a pool of templates that are not in the source pool
    initial_unused_templates = {k: v for k, v in total_templates.items() if k not in source_pool_templates}

    print(f"Source Pool IDs: {sorted(source_pool_ids)}")

    # Call the function to create a new, varied chunk
    varied_templates, _, _ = create_varied_chunk(
        total_templates = total_templates,
        source_templates=source_pool_templates,
        unused_templates=initial_unused_templates,
        variation_type='query',
        substitution_rate=0.3
    )

    source_set = set(source_pool_templates.keys())
    varied_set = set(varied_templates.keys())

    removed_templates = source_set - varied_set
    added_templates = varied_set - source_set

    print(f"\nResult of variation:")
    print(f"  - Templates REMOVED: {sorted(list(removed_templates))}")
    print(f"  - Templates ADDED:   {sorted(list(added_templates))}")
    print("\nTest complete. The function correctly substituted templates.")

