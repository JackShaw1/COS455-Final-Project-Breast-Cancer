#!/bin/bash
#SBATCH --job-name=run_simulations # Job name (will be overridden by --job-name in launch script)
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=64          # CPU cores per task
#SBATCH --mem=64G                  # Memory per node
#SBATCH --time=03:00:00            # Time limit (HH:MM:SS)
#SBATCH --output=output_%j.log     # Output file (%j = job ID)


source .venv/bin/activate

# Activate virtual environment if you have one
# source /path/to/your/venv/bin/activate

# Check if MULTIPLIER is set from the --export flag
# If not, default to 10 so the script can still be run manually
if [ -z "$MULTIPLIER" ]; then
  echo "MULTIPLIER environment variable not set, using default of 10"
  MULTIPLIER=10
fi

echo "Job started on $(hostname)"
echo "Running simulation for ${MULTIPLIER}x multiplier (Job ID: $SLURM_JOB_ID)"

# Run the python script *once* with the specified multiplier
python cli_data_simulation.py --multiplier $MULTIPLIER > ${MULTIPLIER}x_augment.log

echo "Job finished"