#!/bin/bash
#SBATCH -c 4                          # Request four cores
#SBATCH -t 0-03:00                     # Runtime in D-HH:MM format
##SBATCH -p gpu
#SBATCH -p gpu_yu                   # Partition to run in
#SBATCH --gres=gpu:1                           # Number of GPUS
##SBATCH --account=yu_ky98              # with gpu
#SBATCH --account=yu_ky98_contrib     # with gpu_yu
#SBATCH --mem=84G                      # Memory total in MiB (for all cores)
#SBATCH -o logs/immune_benchmark_%a_%j_%N.log
#SBATCH -e logs/immune_benchmark_%a_%j_%N.log

## === PARAMETERS ===
module load gcc/6.2.0 cuda/11.2 miniconda3/23.1.0
source activate /n/data2/hms/dbmi/kyu/lab/bal753/miniconda3/envs/grandqc

SLIDE_FOLDER="/home/sul084/orion_slides"
OUTPUT_DIR="/home/sul084/orion_output"

python wsi_tis_detect.py --slide_folder "$SLIDE_FOLDER" --output_dir "$OUTPUT_DIR" --mpp 0.325

python main.py --slide_folder "$SLIDE_FOLDER" --output_dir "$OUTPUT_DIR"

echo "All processes completed."
