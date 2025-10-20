
import json
import logging
from balance.chunk_segmenter import segment_workloads

# Configure logging to match the segmenter's output
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    # 1. Read and flatten the generated stream
    try:
        with open('workload_stream_query_variation.json', 'r') as f:
            stream_of_chunks = json.load(f)
    except FileNotFoundError:
        logging.error("Could not find 'workload_stream_query_variation.json'. Please run the generator first.")
        return

    # Flatten the list of chunks into a single list of workloads
    # and convert each workload dict into a set of its query texts (templates)
    flat_workload_stream = [set(workload.keys()) for chunk in stream_of_chunks for workload in chunk]

    logging.info(f"--- Successfully loaded and flattened the stream. Total workloads: {len(flat_workload_stream)} ---")

    # 2. Run the segmentation logic
    # Using a 50% threshold as discussed
    difference_threshold = 50
    final_chunks = segment_workloads(flat_workload_stream, difference_threshold)

    print("\n========== Segmentation Complete ==========")
    logging.info(f"Identified {len(final_chunks)} chunks using a {difference_threshold}% threshold.")

    # 3. Analyze and report the differences
    if not final_chunks:
        logging.warning("No chunks were identified by the segmentation logic.")
    else:
        # Get all unique templates from the first chunk
        first_chunk_templates = set().union(*final_chunks[0])
        print(f"\n--- Base templates from Chunk 1 ({len(first_chunk_templates)} unique templates) ---")

        # Compare subsequent chunks to the first one
        for i, chunk in enumerate(final_chunks[1:]):
            chunk_number = i + 2
            current_chunk_templates = set().union(*chunk)

            # Find templates in the current chunk that were NOT in the first chunk
            varied_queries = current_chunk_templates - first_chunk_templates

            print(f"\n--- Varied queries in Chunk {chunk_number} (not present in Chunk 1) ---")
            if varied_queries:
                for query in sorted(list(varied_queries)):
                    print(f"  - {query}")
            else:
                print("  - No new templates found.")

if __name__ == "__main__":
    main()
