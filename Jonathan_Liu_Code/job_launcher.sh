#!/bin/bash

echo "Submitting job for 20x multiplier..."
# We use --export to pass the MULTIPLIER variable to the sbatch script
sbatch --export=ALL,MULTIPLIER=20 --job-name=sim_20x data_augment.sbatch

echo "Submitting job for 50x multiplier..."
sbatch --export=ALL,MULTIPLIER=50 --job-name=sim_50x data_augment.sbatch

echo "Submitting job for 100x multiplier..."
sbatch --export=ALL,MULTIPLIER=100 --job-name=sim_100x data_augment.sbatch

echo "All jobs submitted. Check status with 'squeue -u $USER'"
squeue -u $USER