
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    # 1. Read the generated stream of chunks
    try:
        with open('workload_stream_query_variation.json', 'r') as f:
            stream_of_chunks = json.load(f)
    except FileNotFoundError:
        logging.error("Could not find 'workload_stream_query_variation.json'. Please run the generator first.")
        return

    logging.info(f"--- Successfully loaded a stream of {len(stream_of_chunks)} chunks. ---")

    # 2. Analyze and report the differences between the generated chunks
    if not stream_of_chunks:
        logging.warning("Stream is empty.")
        return

    # Get all unique templates from the first chunk
    # Each workload is a dict of {query: freq}, so we get the keys
    first_chunk_templates = set().union(*(set(workload.keys()) for workload in stream_of_chunks[0]))
    print(f"\n--- Base templates from Generator's Chunk 1 ({len(first_chunk_templates)} unique templates) ---")

    # Compare subsequent chunks to the first one
    for i, chunk in enumerate(stream_of_chunks[1:]):
        chunk_number = i + 2
        current_chunk_templates = set().union(*(set(workload.keys()) for workload in chunk))

        # Find templates in the current chunk that were NOT in the first chunk
        varied_queries = current_chunk_templates - first_chunk_templates

        print(f"\n--- Varied queries in Generator's Chunk {chunk_number} (not present in Chunk 1) ---")
        if varied_queries:
            for query in sorted(list(varied_queries)):
                print(f"  - {query}")
        else:
            print("  - No new templates found.")

if __name__ == "__main__":
    main()
