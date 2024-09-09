# cropRafts.py - Locates and crops individual rafts for image analysis.

from pathlib import Path
import os
from scipy.io import loadmat

from skimage import io, filters, exposure, util
import numpy as np
from PIL import Image


def cropRafts(exp):
    # Initialize numRafts and channels.
    numRafts = exp.arrDim[2] ** 2
    channels = exp.channels

    # Load Loc_cell.mat (links local/global indices) and Centroid_cell.mat (raft px location within subsection).
    loc_cell2 = loadmat(Path(exp.dataPath / 'data' / 'Loc_cell2.mat'))
    loc_cell2 = loc_cell2['Loc_cell2']
    centroid_cell = loadmat(Path(exp.dataPath) / 'data' / 'Centroid_cell.mat')
    centroid_cell = centroid_cell['Centroid_cell']

    # Set up main output folder for individual cropped raft images.
    outPath_main = Path(f'{exp.dataPath}_Analysis') / 'Individual Raft Images'
    os.makedirs(outPath_main, exist_ok=True)

    # Loop through loc_cell for each raft and locate corresponding array subsection folder.
    for arr in range(exp.numArrs):
        for tp in range(exp.numTimePoints):
            for ch in channels:
                raftCount = 0
                for raft in range(0, numRafts):
                    # Identify imaging row/col and local index.
                    img_row = loc_cell2[arr, tp][raft, 1] - 1
                    img_col = loc_cell2[arr, tp][raft, 2] - 1
                    loc_idx = loc_cell2[arr, tp][raft, 3] - 1

                    # Check if raft found (first col = 1) and if raft location in centroid_cell.
                    if loc_cell2[arr, tp][raft, 0] != 0 and loc_idx < len(centroid_cell[arr, tp][img_row, img_col]):
                        raftCount += 1
                        raftCrop = crop(exp, img_row, img_col, loc_idx, centroid_cell, arr, tp, ch)

                        # Save cropped image to subfolders for each specific Array 'arr', Time Point 'tp', and Channel 'ch'.
                        outPath = Path(outPath_main / f'array{arr + 1}_timepoint{tp:02}_{ch}')
                        os.makedirs(outPath, exist_ok=True)
                        raftCrop.save(Path(outPath /
                                           f'array{arr + 1}_'
                                           f'timepoint{tp:02}_'
                                           f'{ch}_'
                                           f'raft{(raft + 1):03}.tif'),
                                       'TIFF')
                        raftCrop.close()

                print(f'(array {arr + 1}) Cropped '
                      f'{raftCount} rafts '
                      f'at Time Point {tp:03} '
                      f'for Channel {ch}.')

    print('Crop complete.')


# Loop through loc_cell for each raft and locate corresponding array subsection folder.
def crop(exp, img_row, img_col, loc_idx, centroid_cell, arr, tp, ch):
    # Identify local px coordinate for specific raft.
    raft_loc_x = centroid_cell[arr, tp][img_row, img_col][loc_idx][0]
    raft_loc_y = centroid_cell[arr, tp][img_row, img_col][loc_idx][1]

    # Locate subsection folder and specific image file for Time Point 'tp' and Channel 'ch'.
    imgPath = Path(exp.dataPath /
                    f'{arr + 1}' /
                    f'{img_row + 1}x{img_col + 1}' /
                    f'{tp:02}_{ch}.tif')
    img = io.imread(imgPath)

    # Crop image to specific raft.
    top = raft_loc_y - (0.89 * (exp.arrDim[0] / 2) / exp.pxSize)
    bottom = raft_loc_y + (0.89 * (exp.arrDim[0] / 2) / exp.pxSize)
    left = raft_loc_x - (0.89 * (exp.arrDim[0] / 2) / exp.pxSize)
    right = raft_loc_x + (0.89 * (exp.arrDim[0] / 2) / exp.pxSize)

    raftCrop = img[int(top):int(bottom), int(left):int(right)]
    raftCrop = Image.fromarray(raftCrop)
    return raftCrop