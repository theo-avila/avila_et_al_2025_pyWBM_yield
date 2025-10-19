#!/bin/bash -l
#SBATCH --job-name=9_weather_only_spatial
#SBATCH --output=9_weather_only_spatial.log
#SBATCH --partition=open
#SBATCH --nodes=1                
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1     
#SBATCH --mem=250GB                
#SBATCH --time=24:00:00

echo "Sourcing Conda"
source $HOME/mambaforge/etc/profile.d/conda.sh || { echo "Failed to source conda.sh"; exit 1; }
echo "Activating WFPIenv"
conda activate pyWBM || { echo "Failed to activate pyWBM env"; exit 1; }

echo "Running Python script"
python /storage/home/cta5244/work/avila_et_al_2025_pyWBM_yield/9_weather_only_spatial.py || { echo "Python script execution failed"; exit 1; }
echo "Job completed"

