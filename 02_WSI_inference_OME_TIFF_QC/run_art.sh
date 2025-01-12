#!/bin/bash
# setting
# SLIDE_FOLDER="/mnt/data2/bal753/Orion/CRC11" # "/path/to/the/slides/"
# OUTPUT_DIR="~/CRC11" # "/path/to/the/result/"

SLIDE_FOLDER="/home/sul084/test"
OUTPUT_DIR="/home/sul084/test"

python wsi_tis_detect.py --slide_folder "$SLIDE_FOLDER" --output_dir "$OUTPUT_DIR"

python main.py --slide_folder "$SLIDE_FOLDER" --output_dir "$OUTPUT_DIR"

echo "All processes completed."
