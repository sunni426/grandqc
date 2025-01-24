#!/bin/bash
#SBATCH -c 4                          # Request four cores
#SBATCH -t 0-03:00                     # Runtime in D-HH:MM format
##SBATCH -p gpu
#SBATCH -p gpu_yu                   # Partition to run in
#SBATCH --gres=gpu:1                           # Number of GPUS
##SBATCH --account=yu_ky98              # with gpu
#SBATCH --account=yu_ky98_contrib     # with gpu_yu
#SBATCH --mem=44G                      # Memory total in MiB (for all cores)
#SBATCH -o logs/immune_benchmark_%a_%j_%N.log
#SBATCH -e logs/immune_benchmark_%a_%j_%N.log



# setting
# SLIDE_FOLDER="/mnt/data2/bal753/Orion/CRC11" # "/path/to/the/slides/"
# OUTPUT_DIR="~/CRC11" # "/path/to/the/result/"

# SLIDE_FOLDER="/home/sul084/test_slides"
# OUTPUT_DIR="/home/sul084/test_output"

SLIDE_FOLDER="/home/sul084/orion_slides"
OUTPUT_DIR="/home/sul084/orion_output"

python wsi_tis_detect.py --slide_folder "$SLIDE_FOLDER" --output_dir "$OUTPUT_DIR"

python main.py --slide_folder "$SLIDE_FOLDER" --output_dir "$OUTPUT_DIR"

echo "All processes completed."
