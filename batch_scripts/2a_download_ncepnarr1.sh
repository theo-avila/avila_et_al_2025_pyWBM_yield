#!/bin/bash -l
#SBATCH --job-name=download_merge_nldas
#SBATCH --output=download_merge_nldas_%A_%a.log
#SBATCH --partition=pches
#SBATCH --array=0-9 
#SBATCH --nodes=1                
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30GB
#SBATCH --time=48:00:00

# Load modules
module load aria2    # For parallel downloads
module load nco      # For merging with ncrcat
module load cdo      # Alternative merging if needed

# Define paths
URL_LIST="/storage/home/cta5244/work/pyWBM_yield_data/NCEPNARR_NLDAS_Hist_Temp/subset_NLDAS_FORA0125_H_002_20250220_223623_.txt"
OUTDIR="/storage/work/cta5244/pyWBM_yield_data/NCEPNARR_NLDAS_Hist_Temp"
# Give each array task its own download directory:
TMPDIR="${OUTDIR}/tmp_downloads_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"

# --- Split URL list among array tasks ---
TOTAL_LINES=$(wc -l < "$URL_LIST")
NUM_TASKS=${SLURM_ARRAY_TASK_COUNT}
LINES_PER_TASK=$(( (TOTAL_LINES + NUM_TASKS - 1) / NUM_TASKS ))
START_LINE=$(( SLURM_ARRAY_TASK_ID * LINES_PER_TASK + 1 ))
END_LINE=$(( START_LINE + LINES_PER_TASK - 1 ))
CHUNK_FILE="${TMPDIR}/urls_${SLURM_ARRAY_TASK_ID}.txt"
sed -n "${START_LINE},${END_LINE}p" "$URL_LIST" > "$CHUNK_FILE"

echo "Task ${SLURM_ARRAY_TASK_ID}: Starting downloads at $(date)"
echo "Processing lines ${START_LINE} to ${END_LINE} from $URL_LIST"

# Use aria2 to download URLs in parallel
aria2c -j 10 -x 10 -s 10 --input-file="$CHUNK_FILE" --dir="$TMPDIR"

echo "Task ${SLURM_ARRAY_TASK_ID}: Downloads complete at $(date)"
echo "Task ${SLURM_ARRAY_TASK_ID}: Renaming files with long names if needed..."

# Rename files with long names by extracting a shorter name from the LABEL parameter
for f in "$TMPDIR"/HTTP_services.cgi*; do
  NEWNAME=$(echo "$f" | sed -n 's/.*[?&]LABEL=\([^&]*\).*/\1/p')
  if [ -n "$NEWNAME" ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID}: Renaming $f to ${TMPDIR}/${NEWNAME}"
    mv "$f" "${TMPDIR}/${NEWNAME}"
  fi
done

echo "Task ${SLURM_ARRAY_TASK_ID}: Renaming complete."
echo "Task ${SLURM_ARRAY_TASK_ID}: Starting per–task merging by year..."

# --- Parallel merging by year within this task ---
# Look for downloaded NetCDF files and extract unique years
years=$(ls "$TMPDIR"/NLDAS_FORA0125_H.A*.nc4 2>/dev/null | grep -oP 'A\K\d{4}' | sort -u)
for year in $years; do
    echo "Task ${SLURM_ARRAY_TASK_ID}: Merging files for year $year..."
    # Each task writes its own merged file per year
    ncrcat "$TMPDIR"/NLDAS_FORA0125_H.A${year}*.nc4 "${OUTDIR}/NLDAS_FORA_${year}_${SLURM_ARRAY_TASK_ID}.nc" &
done
wait
echo "Task ${SLURM_ARRAY_TASK_ID}: Per–task merging complete at $(date)"
