import argparse
from pathlib import Path
import sys

import tensorflow as tf
import wandb


def process_and_log_run(log_dir: Path, project_name: str, entity: str = None):
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
    new_run = wandb.init(project=project_name, entity=entity, name=run_name, reinit=True)

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
        description="Re-log TensorBoard experiments to W&B without folder prefixes."
    )
    parser.add_argument("--path", type=str, default='', help="Root directory containing TensorBoard runs.")
    parser.add_argument("--proj_name", type=str, default="My_Flattened_Grid_Search", help="W&B project name.")
    parser.add_argument("--entity", type=str, default=None, help="W&B entity/user.")
    parser.add_argument("--start_id", type=int, help="Optional starting experiment ID (inclusive).")
    parser.add_argument("--stop_id", type=int, help="Optional stopping experiment ID (inclusive).")
    parser.add_argument("--run_dir", action="append", help="Explicit run directory to sync (can be passed multiple times).")

    args = parser.parse_args()

    root_log_dir = Path(args.path or "gridsearch/")
    if args.run_dir:
        target_dirs = [Path(d) for d in args.run_dir]
    elif args.start_id is not None and args.stop_id is not None:
        target_dirs = []
        for i in range(args.start_id, args.stop_id + 1):
            pattern = f"tblog_ID*_{i}"
            matches = list(root_log_dir.glob(pattern))
            if matches:
                target_dirs.append(matches[0])
            else:
                print(f"⚠️ Warning: Directory matching pattern '{pattern}' not found. Skipping ID {i}.")
    else:
        target_dirs = sorted([p for p in root_log_dir.iterdir() if p.is_dir() and p.name.startswith("tblog_")])

    if not target_dirs:
        print("Error: No experiment directories matched the specified criteria. Exiting.")
        sys.exit(1)

    print(f"Found {len(target_dirs)} directories to process.")

    for directory in target_dirs:
        process_and_log_run(directory, args.proj_name, args.entity)

    print("\nAll requested directories processed.")


if __name__ == "__main__":
    main()
