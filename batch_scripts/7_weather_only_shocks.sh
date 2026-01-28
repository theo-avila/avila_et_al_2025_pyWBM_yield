#!/bin/bash -l
#SBATCH --job-name=7_weather_only_shocks
#SBATCH --output=7_weather_only_shocks.log
#SBATCH --partition=open
#SBATCH --nodes=1                
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1     
#SBATCH --mem=200GB                
#SBATCH --time=24:00:00

echo "Sourcing Conda"
source $HOME/mambaforge/etc/profile.d/conda.sh || { echo "Failed to source conda.sh"; exit 1; }
echo "Activating pyWBM"
conda activate pyWBM || { echo "Failed to activate pyWBM env"; exit 1; }

echo "Running Python script"
python /storage/home/cta5244/work/avila_et_al_2025_pyWBM_yield/7_weather_only_shocks.py || { echo "Python script execution failed"; exit 1; }
echo "Job completed"

