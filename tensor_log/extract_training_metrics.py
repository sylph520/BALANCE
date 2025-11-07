#!/usr/bin/env python3
import argparse
import os
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

def extract_training_metrics(logdir, output_csv, downsample=10):
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()

    all_tags = ea.Tags()['scalars']
    print("Available scalar tags in log:", all_tags)

    # Updated mapping for your SB2 log
    key_tags = [
        'episode_reward',                   # episode reward
        'loss/policy_gradient_loss',        # policy loss
        'loss/value_function_loss',         # value loss
        'loss/entropy_loss',                # entropy
        'loss/approximate_kullback-leibler',# approx KL
        'loss/clip_factor',                 # clip fraction
        'train/explained_variance'          # explained variance
    ]

    selected_tags = [tag for tag in key_tags if tag in all_tags]
    if not selected_tags:
        raise ValueError("None of the expected PPO2 metrics found in this log. Available tags:\n" + str(all_tags))

    print("Selected tags for extraction:", selected_tags)

    data_dict = {}
    for tag in selected_tags:
        events = ea.Scalars(tag)
        df = pd.DataFrame({
            'step': [e.step for e in events],
            'value': [e.value for e in events]
        })
        df = df.iloc[::downsample, :]
        df.rename(columns={'value': tag.replace('/', '_')}, inplace=True)
        data_dict[tag] = df

    # Merge all metrics on 'step'
    merged_df = None
    for df in data_dict.values():
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on='step', how='outer')

    merged_df.sort_values('step', inplace=True)
    merged_df.to_csv(output_csv, index=False)
    print(f"Combined training metrics saved to: {output_csv}")

def main():
    parser = argparse.ArgumentParser(
        description="Extract PPO2 training metrics for analysis to improve training."
    )
    parser.add_argument('--logdir', type=str, required=True, help='TensorBoard log directory')
    parser.add_argument('--output', type=str, default='ppo2_training_metrics.csv', help='Output CSV file')
    parser.add_argument('--downsample', type=int, default=10, help='Keep every Nth scalar to reduce CSV size')
    args = parser.parse_args()

    if not os.path.exists(args.logdir):
        raise FileNotFoundError(f"Log directory not found: {args.logdir}")

    extract_training_metrics(args.logdir, args.output, args.downsample)

if __name__ == "__main__":
    main()

