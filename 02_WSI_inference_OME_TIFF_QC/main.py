"""
Comments to version:
- Uses tissue maps from tissue detector. Therefore, slides should be processed by tissue detector firstly.
- Consider adding color schema if you use the tool for a new entity
"""
from wsi_colors import colors_QC7 as colors
import torch
from PIL import Image
import os
from wsi_slide_info import slide_info
from wsi_process import slide_process_single
from wsi_maps import make_overlay
import numpy as np
import timeit
import cv2
import zarr
import tifffile
from skimage.io import imread
from tqdm import tqdm
import argparse
# Image.MAX_IMAGE_PIXELS = 1000000000
Image.MAX_IMAGE_PIXELS = None

# DEVICE
DEVICE = 'cuda'
'''
'cuda:0' - NVIDIA GPU card
'mps'    - APPLE Silicon
'''

# MODEL(S)
# MODEL 1: Artifacts detection

MODEL_TUMOR_DIR = './models/qc/'
MODEL_TUMOR_NAME = 'GrandQC_MPP15.pth'
MPP_MODEL_1 = 1.5
M_P_S_MODEL_1 = 512
ENCODER_MODEL_1 = 'timm-efficientnet-b0'
ENCODER_MODEL_1_WEIGHTS = 'imagenet'
BACK_CLASS = 7

parser = argparse.ArgumentParser()
parser.add_argument('--slide_folder', dest='slide_folder', help='path to WSIs', type=str)
parser.add_argument('--output_dir', dest='output_dir', help='path to output folder', type=str)
parser.add_argument('--start', dest='start', default=0, help='start num of WSIs', type=int)
parser.add_argument('--end', dest='end', default=-1, help='end num of WSIs', type=int)
parser.add_argument('--ol_factor', dest='ol_factor', default=10,
                    help='reduction factor of the overlay compared to dimensions of original WSI', type=int)

args = parser.parse_args()

start = args.start
end = args.end
SLIDE_DIR = args.slide_folder
OUTPUT_DIR = args.output_dir
OVERLAY_FACTOR = args.ol_factor

if end == -1:
    end = len(os.listdir(SLIDE_DIR))

# OUTPUT (TEXT)
REPORT_FILE_NAME = 'report_slides_' + str(start) + "_" + str(end)   # File name, ".txt" will be added in the end.
REPORT_OUTPUT_DIR = OUTPUT_DIR  # where to save the text report
obj_power = 99

# =============================================================================
# LOAD MODELS
# =============================================================================
model = torch.load(MODEL_TUMOR_DIR + MODEL_TUMOR_NAME, map_location=DEVICE)

# ====================================================================
# PREPARE REPORT FILE, OUTPUT FOLDERS
# =============================================================================

# Prepare report file header
path_result = os.path.join(REPORT_OUTPUT_DIR, REPORT_FILE_NAME + "_stats_per_slide.txt")
output_header = "slide_name" + "\t" + "obj_power" + "\t" + "mpp" + "\t"
output_header = output_header + "patch_n_h_l0" + "\t" + "patch_n_w_l0" + "\t"
output_header = output_header + "patch_overall" + "\t"
output_header = output_header + "width" + "\t" + "height" + "\t"
output_header = output_header + "Time"
output_header = output_header + "\n"
results = open(path_result, "a+")
results.write(output_header)
results.close()

maps_dir = os.path.join(OUTPUT_DIR, 'maps_qc')
overlay_dir = os.path.join(OUTPUT_DIR, 'overlays_qc')
mask_dir = os.path.join(OUTPUT_DIR, 'mask_qc')

# try:
#     os.makedirs(maps_dir)
#     os.makedirs(overlay_dir)
#     os.makedirs(mask_dir)
# except:
#     print('The target folders are already there ..')

dirs_to_create = [maps_dir, overlay_dir, mask_dir]

for directory in dirs_to_create:
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            print(f"Directory created: {directory}")
        except Exception as e:
            print(f"Error creating directory {directory}: {e}")
    else:
        print(f"Directory already exists: {directory}")

# ====================================================================
# MAIN SCRIPT
# =============================================================================

# Read in slide names
# slide_names = sorted([f for f in os.listdir(SLIDE_DIR) if not f.startswith('.')])
slide_names = ["19510_C11_US_SCAN_OR_001__151039-registered.ome.tif"]

# Start analysis loop
for slide_name in slide_names[start:end]:
    # Register start time
    start = timeit.default_timer()
    # Open slide
    path_slide = os.path.join(SLIDE_DIR, slide_name)
    print(f'path_slide: {path_slide}')
    # slide_tiff = imread(path_slide, aszarr=True) # Error - OSError: ImageIO does not generally support reading folders.
    # slide_tiff = Image.open(path_slide)  # Use PIL to open the image
    # slide_original = zarr.open(slide_tiff, mode='r')
    # slide = slide_original[0]

    # ALTERNATIVE
    slide_tiff = tifffile.imread(path_slide)  # Read image into NumPy array
    zarr_store = zarr.open('path_to_zarr_store', mode='w', shape=slide_tiff.shape, dtype=slide_tiff.dtype)
    zarr_store[:] = slide_tiff
    slide_original = zarr.open('path_to_zarr_store', mode='r')
    slide = slide_original[:]  # Read all data from the Zarr store

    # Transpose, Sunni 1/9
    slide = np.transpose(slide, (2, 0, 1))
    print(f'SLIDE INFO {slide.shape}')
    
    # GET SLIDE INFO
    p_s, patch_n_w_l0, patch_n_h_l0, mpp, w_l0, h_l0 = slide_info(slide, M_P_S_MODEL_1, MPP_MODEL_1)

    # LOAD TISSUE DETECTION MAP
    try:
        tis_det_map = Image.open(OUTPUT_DIR + "/tis_det_mask/" + slide_name + '_MASK.png')
        '''
        Tissue detection map is generated on MPP = 10
        This map is used for on-fly control of the necessity of model inference.
        Two variants: reduced version with perfect correlation or full version scaled to working MPP of the tumor detection model
        Classes: 0 - tissue, 1 - background
        '''
        tis_det_map_mpp = np.array(tis_det_map.resize((int(w_l0 * mpp / MPP_MODEL_1),
                                                       int(h_l0 * mpp / MPP_MODEL_1)), Image.Resampling.LANCZOS))
    except Exception as e:
        print(f"Loading tissue detection map issue, initialize map as all 0's. {e}")
        tis_det_map_mpp = np.zeros((int(w_l0 * mpp / MPP_MODEL_1), int(h_l0 * mpp / MPP_MODEL_1)))

    try:
        map, full_mask = slide_process_single(model, tis_det_map_mpp, slide, patch_n_w_l0, patch_n_h_l0, p_s,
                                              M_P_S_MODEL_1, colors, ENCODER_MODEL_1,
                                              ENCODER_MODEL_1_WEIGHTS, DEVICE, BACK_CLASS)
    except Exception as e:
        print(f"Something wrong with processing: {e}")
        continue
    stop = timeit.default_timer()
    map_path = os.path.join(maps_dir, slide_name + "_map_ALL.png")
    map.save(map_path)

    mask_path = os.path.join(mask_dir, slide_name + "_mask.png")
    cv2.imwrite(mask_path, full_mask)

    # =============================================================================
    # 8. MAKE AND SAVE OVERLAY for C8: HEATMAP ON REDUCED AND CROPPED SLIDE CLON
    # =============================================================================
    # overlay = make_overlay(slide_original, map, p_s, patch_n_w_l0, patch_n_h_l0, OVERLAY_FACTOR) # og
    overlay = make_overlay(slide, map, p_s, patch_n_w_l0, patch_n_h_l0, OVERLAY_FACTOR) # slide_original --> slide, Sunni

    # Save overlaid image
    overlay_im = Image.fromarray(overlay)
    overlay_im_name = os.path.join(overlay_dir, slide_name + "_overlay_QC.jpg")
    overlay_im.save(overlay_im_name)

    # Write down per slide result
    # Basic data about slide (size, pixel size, objective power, height, width)
    output_temp = slide_name + "\t"
    output_temp = output_temp + str(obj_power) + "\t" + str(mpp) + "\t"
    output_temp = output_temp + str(patch_n_h_l0) + "\t" + str(patch_n_w_l0) + "\t"
    output_temp = output_temp + str(patch_n_h_l0 * patch_n_w_l0) + "\t"
    output_temp = output_temp + str(patch_n_w_l0 * p_s) + "\t" + str(patch_n_h_l0 * p_s) + "\t"
    output_temp = output_temp + str(round((stop - start) / 60, 1))
    output_temp = output_temp + "\n"

    results = open(path_result, "a+")
    results.write(output_temp)
    results.close()
    # slide_tiff.close()
