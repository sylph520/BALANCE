
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def analyze_stream(filename):
    """Analyzes a given workload stream file and prints the variations."""
    try:
        with open(filename, 'r') as f:
            stream_of_chunks = json.load(f)
    except FileNotFoundError:
        logging.error(f"Could not find '{filename}'. Please run the generator first.")
        return

    logging.info(f"--- Successfully loaded a stream of {len(stream_of_chunks)} chunks from {filename}. ---")

    if not stream_of_chunks:
        logging.warning("Stream is empty.")
        return

    # Get all unique templates from the first chunk
    first_chunk_templates = set().union(*(set(workload.keys()) for workload in stream_of_chunks[0]))
    print(f"\n--- Base templates from Generator's Chunk 1 ({len(first_chunk_templates)} unique templates) ---")

    # Compare subsequent chunks to the first one
    for i, chunk in enumerate(stream_of_chunks[1:]):
        chunk_number = i + 2
        current_chunk_templates = set().union(*(set(workload.keys()) for workload in chunk))

        varied_queries = current_chunk_templates - first_chunk_templates

        print(f"\n--- Varied queries in Generator's Chunk {chunk_number} (not present in Chunk 1) ---")
        if varied_queries:
            for query in sorted(list(varied_queries)):
                print(f"  - {query}")
        else:
            print("  - No new templates found.")

def main():
    variation_types = ['query', 'frequency', 'combined']
    for var_type in variation_types:
        filename = f'workload_stream_{var_type}_variation.json'
        print(f"\n=================================================================")
        print(f"    ANALYZING STREAM: '{var_type.upper()}' VARIATION")
        print(f"=================================================================")
        analyze_stream(filename)

if __name__ == "__main__":
    main()
