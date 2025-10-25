import logging
from .query_hasher import query_to_hash

# Configure logging for clear output
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def segment_workloads(workload_stream, difference_threshold_x, hash2tid: dict, dbname: str):
    """
    Segments a stream of workloads into chunks based on template differences.

    Args:
        workload_stream (list of sets): A list where each element is a set of 
                                        query template IDs for that workload.
        difference_threshold_x (int): The X% threshold. If the percentage of new
                                      templates in an incoming workload is less
                                      than this, it's merged.

    Returns:
        list of lists: A list of chunks, where each chunk is a list of indices to the original stream.
    """
    if not workload_stream:
        return []
    workload_stream_ids = []
    for i in workload_stream:
        w_tids = set()
        for j in i:
            h, j = query_to_hash(j, dbname=dbname, unify_similar_ops=True)
            tid =  hash2tid[h]
            w_tids.add(tid)
        workload_stream_ids.append(w_tids)
    workload_stream = workload_stream_ids
    chunks = []
    # Start the first chunk with the first workload
    current_chunk_templates = set([i for i in workload_stream[0]])
    start_index = 0
    
    logging.info(f"Starting Chunk 1 with workload: {workload_stream[0]}")

    # Iterate over the rest of the workloads
    for i, next_workload_templates in enumerate(workload_stream[1:]):
        workload_number = i + 2
        logging.info(f"--- Analyzing Workload {workload_number} ({next_workload_templates}) ---")

        if not next_workload_templates:
            logging.warning(f"Workload {workload_number} is empty, merging into current chunk.")
            continue

        # Calculate the set of new templates in the incoming workload
        new_templates = next_workload_templates - current_chunk_templates
        
        # Calculate the difference percentage
        diff_percent = (len(new_templates) / len(next_workload_templates)) * 100
        
        logging.info(f"Templates in current chunk: {current_chunk_templates}")
        logging.info(f"New templates in workload {workload_number}: {new_templates}")
        logging.info(f"Difference percentage: {diff_percent:.2f}%")

        if diff_percent < difference_threshold_x:
            # Merge into the current chunk
            logging.info(f"Difference is less than {difference_threshold_x}%. Merging into current chunk.")
            current_chunk_templates.update(next_workload_templates)
        else:
            # Difference is too high, start a new chunk
            logging.info(f"Difference is >= {difference_threshold_x}%. Finalizing current chunk and starting a new one.")
            chunks.append(list(range(start_index, i + 1)))
            start_index = i + 1
            
            # Start the new chunk
            current_chunk_templates = set(next_workload_templates)
            logging.info(f"Starting Chunk {len(chunks) + 1} with workload: {next_workload_templates}")

    # Add the last chunk to the list of chunks
    chunks.append(list(range(start_index, len(workload_stream))))
    logging.info("--- Stream finished. Finalizing the last chunk. ---")

    return chunks
if __name__ == "__main__":
    print("##### TEST SCENARIO: 4 Identical Static Workloads #####")
    # Define the scenario: 4 identical workloads arriving one by one.
    # Each workload consists of the same three query templates.
    static_workload_stream = [
        {"q1", "q5", "q8"},  # Workload 1
        {"q1", "q5", "q8"},  # Workload 2
        {"q1", "q5", "q8"},  # Workload 3
        {"q1", "q5", "q8"},  # Workload 4
    ]

    # Set a difference threshold of 20% as used in the paper.
    threshold = 20

    print(f"\nInput Stream: {static_workload_stream}")
    print(f"Difference Threshold: {threshold}%\n")

    # Run the segmentation logic
    final_chunks = segment_workloads(static_workload_stream, threshold)

    # Print the results
    print("\n##### RESULTS #####")
    print(f"Total number of chunks identified: {len(final_chunks)}")
    for i, chunk in enumerate(final_chunks):
        print(f"  Chunk {i+1} contains {len(chunk)} workloads: {chunk}")

    # Verify the result for the test case
    assert len(final_chunks) == 1, "Test Failed: Expected 1 chunk!"
    assert len(final_chunks[0]) == 4, "Test Failed: Expected 4 workloads in the chunk!"
    print("\nAssertion PASSED: As expected, all 4 identical workloads were classified into a single chunk.")
