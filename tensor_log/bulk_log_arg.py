import tensorflow as tf
from pathlib import Path
import wandb
import re
import argparse
import sys

# --- Configuration ---
# The root folder containing all your experiment subdirectories
ROOT_LOG_DIR = Path("gridsearch/")
# Your desired W&B project name (all runs will be grouped here)
WANDB_PROJECT_NAME = "My_Flattened_Grid_Search"
# ---------------------

def process_and_log_run(log_dir: Path):
    """
    Reads a single TensorBoard event file, extracts clean metrics, 
    and logs them to a new W&B run.
    """
    
    # 1. Find the event file
    try:
        event_file = next(log_dir.glob('events.out.tfevents.*'))
    except StopIteration:
        print(f"Skipping {log_dir.name}: No 'events.out.tfevents' file found.")
        return

    # 2. Determine the new W&B Run Name
    # Extract the full name and try to clean it up for the W&B Run Name
    run_name = log_dir.name.replace('tblog_', '').strip()
        
    print(f"\n--- Processing Run: {run_name} ---")

    # 3. Extract and organize the data from the event file
    data_by_step = {}
    
    # Use the summary_iterator to read the raw data
    for event in tf.compat.v1.train.summary_iterator(str(event_file)):
        if event.summary.value:
            step = event.step
            if step not in data_by_step:
                data_by_step[step] = {}
            
            for value in event.summary.value:
                # value.tag is the exact metric name (no prefix)
                metric_name = value.tag 
                metric_value = value.simple_value
                
                # Use the raw metric name directly
                clean_name = metric_name
                
                data_by_step[step][clean_name] = metric_value

    # 4. Start a new W&B run
    new_run = wandb.init(project=WANDB_PROJECT_NAME, name=run_name, reinit=True)
    
    # 5. Log the flattened data step-by-step
    for step, metrics in sorted(data_by_step.items()):
        clean_metrics = {k: v for k, v in metrics.items() if v is not None}
        if clean_metrics:
            wandb.log(clean_metrics, step=step)

    # 6. Finalize the run
    new_run.finish()
    print(f"✅ Finished logging {run_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Bulk re-log TensorBoard experiments to W&B without folder prefixes."
    )
    parser.add_argument(
        '--start_id',
        type=int,
        required=True,
        help="The starting experiment ID (inclusive), e.g., 1 for Experiment_1."
    )
    parser.add_argument(
        '--stop_id',
        type=int,
        required=True,
        help="The stopping experiment ID (inclusive), e.g., 32 for Experiment_32."
    )
    
    args = parser.parse_args()

    # Create the list of expected directory names to sync
    expected_dirs = []
    # Loop from start_id up to and including stop_id
    for i in range(args.start_id, args.stop_id + 1):
        # We need to construct a flexible pattern since the unique hash is long and variable.
        # We'll use a glob pattern to match the directory.
        pattern = f"tblog_ID_Test_Experiment_1_Index_Count_TPCDSC_ts100000_dmxsz50_uniFreq_idxmax5OFFprecedentMasking_{i}"

        # Use Path.glob to find the matching directory
        found_dirs = list(ROOT_LOG_DIR.glob(pattern))
        
        if found_dirs:
            # We assume there is only one match per experiment ID
            expected_dirs.append(found_dirs[0])
        else:
            print(f"⚠️ Warning: Directory matching pattern '{pattern}' not found. Skipping ID {i}.")


    if not expected_dirs:
        print("Error: No experiment directories matched the specified ID range. Exiting.")
        sys.exit(1)

    print(f"Found {len(expected_dirs)} directories to process (IDs {args.start_id} to {args.stop_id}).")
    
    # Process the found directories
    for directory in sorted(expected_dirs):
        process_and_log_run(directory)

    print("\nAll requested directories processed.")


if __name__ == "__main__":
    main()
