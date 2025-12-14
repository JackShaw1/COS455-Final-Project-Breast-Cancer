#!/usr/bin/env python3
"""
Combine multiple breast cancer datasets vertically with a dataset identifier column.
"""

import pandas as pd
import sys

def combine_datasets(output_path='combined_breast_datasets.csv'):
    """
    Combine four breast cancer datasets vertically and add a numeric dataset identifier.

    Args:
        output_path: Path where the combined CSV will be saved
    """
    # Define the datasets to combine
    datasets = [
        '/n/fs/vision-mix/jl0796/qcb/qcb455_project/Breast_GSE7904.csv',
        '/n/fs/vision-mix/jl0796/qcb/qcb455_project/Breast_GSE26910.csv',
        '/n/fs/vision-mix/jl0796/qcb/qcb455_project/Breast_GSE42568.csv',
        '/n/fs/vision-mix/jl0796/qcb/qcb455_project/Breast_GSE45827.csv'
    ]

    # List to store dataframes
    dfs = []

    print("Reading datasets...")
    for idx, dataset_path in enumerate(datasets, start=1):
        print(f"  Reading {dataset_path.split('/')[-1]}...")
        df = pd.read_csv(dataset_path)

        # Add dataset identifier column (1, 2, 3, 4)
        df['dataset_id'] = idx

        print(f"    Shape: {df.shape}")
        dfs.append(df)

    # Combine all dataframes vertically
    print("\nCombining datasets...")
    combined_df = pd.concat(dfs, axis=0, ignore_index=True)

    print(f"Combined dataset shape: {combined_df.shape}")
    print(f"Dataset ID distribution:\n{combined_df['dataset_id'].value_counts().sort_index()}")

    # Save to CSV
    print(f"\nSaving to {output_path}...")
    combined_df.to_csv(output_path, index=False)
    print("Done!")

    return combined_df

if __name__ == '__main__':
    # Allow optional output path as command line argument
    output_path = sys.argv[1] if len(sys.argv) > 1 else 'combined_breast_datasets.csv'
    combine_datasets(output_path)
