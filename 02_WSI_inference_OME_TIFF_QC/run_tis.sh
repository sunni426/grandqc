#!/bin/bash
#SBATCH -c 1                          # Request four cores
#SBATCH -t 0-01:00                     # Runtime in D-HH:MM format
#SBATCH -p gpu
##SBATCH -p gpu_yu                   # Partition to run in
#SBATCH --gres=gpu:1                           # Number of GPUS
#SBATCH --account=yu_ky98              # with gpu
##SBATCH --account=yu_ky98_contrib     # with gpu_yu
#SBATCH --mem=2G                      # Memory total in MiB (for all cores)
#SBATCH -o logs/immune_benchmark_%a_%j_%N.log
#SBATCH -e logs/immune_benchmark_%a_%j_%N.log

SLIDE_FOLDER="/home/sul084/test_slides"
OUTPUT_DIR="/home/sul084/test_output"

python wsi_tis_detect.py --slide_folder "$SLIDE_FOLDER" --output_dir "$OUTPUT_DIR"

echo "Tissue Segmentation is completed!"
