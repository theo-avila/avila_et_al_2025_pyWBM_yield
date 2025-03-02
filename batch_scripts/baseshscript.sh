#!/bin/bash -l
#SBATCH --job-name=
#SBATCH --output=
#SBATCH --partition=pches
#SBATCH --nodes=1                
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2         # Adjusted for a driver job; change if necessary.
#SBATCH --mem=30GB                # Adjusted for driver needs.
#SBATCH --time=24:00:00

# Load Conda
echo "Sourcing Conda"
source $HOME/mambaforge/etc/profile.d/conda.sh || { echo "Failed to source conda.sh"; exit 1; }
echo "Activating WFPIenv"
conda activate pyWBM || { echo "Failed to activate pyWBM env"; exit 1; }

# Run the Python script
echo "Running Python script"
python /storage/home/cta5244/work/avila_et_al_2025_pyWBM_yield/2a_data_processing_gdd.py || { echo "Python script execution failed"; exit 1; }
echo "Job completed"

