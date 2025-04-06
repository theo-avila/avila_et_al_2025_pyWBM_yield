#!/bin/bash -l
#SBATCH --job-name=pyWBM_future_dday
#SBATCH --output=hr_day.log
#SBATCH --partition=open
#SBATCH --nodes=1                
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1         
#SBATCH --mem=20GB                
#SBATCH --time=24:00:00

# Load Conda
echo "Sourcing Conda"
source $HOME/mambaforge/etc/profile.d/conda.sh || { echo "Failed to source conda.sh"; exit 1; }
echo "Activating WFPIenv"
conda activate pyWBM || { echo "Failed to activate pyWBM env"; exit 1; }

export PYTHONPATH=/storage/home/cta5244/work/avila_et_al_2025_pyWBM_yield:$PYTHONPATH

# Run the Python script
echo "Running Python script"
python /storage/home/cta5244/work/avila_et_al_2025_pyWBM_yield/6_future_processing.py || { echo "Python script execution failed"; exit 1; }
echo "Job completed"

