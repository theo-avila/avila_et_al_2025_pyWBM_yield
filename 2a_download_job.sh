#!/bin/bash -l
#SBATCH --job-name=download_tas_data
#SBATCH --output=download_tas_data.log
#SBATCH --partition=seseml
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=60GB
#SBATCH --time=2:00:00

module load wget

wget -nd -P /storage/work/cta5244/pyWBM_yield_data/LOCA2_ssp \
  https://cirrus.ucsd.edu/~pierce/LOCA2/NAmer/GFDL-CM4/0p0625deg/r1i1p1f1/ssp245/tasmin/tasmin.GFDL-CM4.ssp245.r1i1p1f1.2015-2044.LOCA_16thdeg_v20220413.nc \
  https://cirrus.ucsd.edu/~pierce/LOCA2/NAmer/GFDL-CM4/0p0625deg/r1i1p1f1/ssp245/tasmax/tasmax.GFDL-CM4.ssp245.r1i1p1f1.2015-2044.LOCA_16thdeg_v20220413.nc

echo "Download complete!"