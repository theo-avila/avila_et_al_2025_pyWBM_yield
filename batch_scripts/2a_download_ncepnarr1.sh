#!/bin/bash -l
#SBATCH --job-name=download_nldas
#SBATCH --output=download_nldas_%j.log
#SBATCH --partition=seseml
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=60GB
#SBATCH --time=24:00:00

module load wget
module load nco      # For NetCDF merging
module load cdo      # Alternative merging (if needed)

# Define paths
URL_LIST="/storage/home/cta5244/work/pyWBM_yield_data/NCEPNARR_NLDAS_Hist_Temp/subset_NLDAS_FORA0125_H_002_20250220_223623_.txt"
OUTDIR="/storage/work/cta5244/pyWBM_yield_data/NCEPNARR_NLDAS_Hist_Temp"
TMPDIR="$OUTDIR/tmp_downloads"
mkdir -p "$TMPDIR"

echo "Starting downloads at $(date)"

wget --auth-no-challenge --content-disposition --directory-prefix="$TMPDIR" --continue --input-file="$URL_LIST"

# Download files using wget
# --content-disposition tells wget to use the server-provided file name (if available)
wget --content-disposition --directory-prefix="$TMPDIR" --continue --input-file="$URL_LIST"

echo "Download complete at $(date)"
echo "Checking for long file names and renaming if needed..."

# Rename files with long names by extracting a shorter name from the LABEL parameter in the URL.
# This loop looks for files that start with "HTTP_services.cgi" (which indicates the long names)
for f in "$TMPDIR"/HTTP_services.cgi*; do
  # Extract the LABEL parameter value (assumes URL contains "&LABEL=desired_name")
  NEWNAME=$(echo "$f" | sed -n 's/.*[?&]LABEL=\([^&]*\).*/\1/p')
  if [ -n "$NEWNAME" ]; then
    echo "Renaming $f to $TMPDIR/$NEWNAME"
    mv "$f" "$TMPDIR/$NEWNAME"
  fi
done

echo "Renaming complete."
echo "Starting yearly merging..."

# Change to temporary directory
cd "$TMPDIR" || exit

# Loop over unique years extracted from file names.
# Assumes files are now named like: NLDAS_FORA0125_H.A19790101.1300.002.grb.SUB.nc4
for YEAR in $(ls NLDAS_FORA0125_H.A*.nc4 2>/dev/null | grep -oP 'A\K\d{4}' | sort -u); do
    echo "Merging files for year $YEAR..."
    # Merge all files from that year into one yearly NetCDF file.
    ncrcat NLDAS_FORA0125_H.A$YEAR*.nc4 "$OUTDIR/NLDAS_FORA_$YEAR.nc"
    # Alternatively, you could use CDO:
    # cdo mergetime NLDAS_FORA0125_H.A$YEAR*.nc4 "$OUTDIR/NLDAS_FORA_$YEAR.nc"
    echo "Merged data for $YEAR completed!"
done

echo "All merging complete at $(date)"

# Optionally remove temporary downloaded files
rm -rf "$TMPDIR"

echo "Processing finished!"
