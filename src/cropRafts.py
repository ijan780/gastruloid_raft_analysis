# cropRafts.py - Locates and crops individual rafts for image analysis.

from pathlib import Path
from scipy.io import loadmat

from skimage import io
import numpy as np
import tifffile as tiff

def cropRafts(exp):
    # Initialize numRafts and channels.
    numRafts = exp.arrDim[2] ** 2
    ch_end = exp.channels_end
    ch_tl = exp.channels_tl

    # Load Loc_cell.mat (links local/global indices) and Centroid_cell.mat (raft px location within subsection).
    loc_cell2 = loadmat(exp.dataPath / 'data' / 'Loc_cell2.mat')
    loc_cell2 = loc_cell2['Loc_cell2']
    centroid_cell = loadmat(exp.dataPath / 'data' / 'Centroid_cell.mat')
    centroid_cell = centroid_cell['Centroid_cell']

    # Set up main output folder for individual cropped raft images.
    outPath_main = Path(f'{exp.dataPath}_Analysis') / 'Individual Raft Images'
    outPath_main.mkdir(exist_ok=True)

    # Loop through loc_cell for each raft and locate corresponding array subsection folder.
    for arr in range(exp.numArrs):
        raftCount = 0
        for raft in range(0, numRafts):
            # Check if raft exists at every time point.
            loc_cell2_raft = [tp[raft] for tp in loc_cell2[1, :]]
            loc_idx_raft = [tp[3] for tp in loc_cell2_raft]
            if 0 in loc_idx_raft:
                continue

            raftCount += 1
            crop_tp = []
            # Crop raft at each time point for all time-lapse channels then stack into single TIFF as (tp, c, y, x)
            for tp in range(exp.numTimePoints):
                # Identify imaging row/col and local index.
                img_row = loc_cell2[arr, tp][raft, 1] - 1
                img_col = loc_cell2[arr, tp][raft, 2] - 1
                loc_idx = loc_cell2[arr, tp][raft, 3] - 1

                crop_ch = []
                for ch in ch_tl:
                    raftCrop = crop(exp, img_row, img_col, loc_idx, centroid_cell, arr, tp, ch)
                    crop_ch.append(raftCrop)
                stacked_ch = np.stack(crop_ch, axis=0)
                crop_tp.append(stacked_ch)
            tl_img = np.stack(crop_tp, axis=0)
            outPath_tl = outPath_main / 'Time-lapse' / f'array{arr + 1}_raft{(raft + 1):03}_tl.tif'
            tiff.imwrite(outPath_tl, tl_img)

            crop_ch = []
            for ch in ch_end:
                raftCrop = crop(exp, img_row, img_col, loc_idx, centroid_cell,
                                arr, (exp.numTimePoints - 1), ch)
                crop_ch.append(raftCrop)
            end_img = np.stack(crop_ch, axis=0)
            outPath_end = outPath_main / 'Endpoint' / f'array{arr + 1}_raft{(raft + 1):03}_end.tif'
            tiff.imwrite(outPath_end, end_img)

        print(f'(array {arr + 1}) Cropped '
              f'{raftCount} rafts ')

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

    with tiff.TiffFile(imgPath) as img:
        # Crop image to specific raft.
        top = raft_loc_y - (0.89 * (exp.arrDim[0] / 2) / exp.pxSize)
        bottom = raft_loc_y + (0.89 * (exp.arrDim[0] / 2) / exp.pxSize)
        left = raft_loc_x - (0.89 * (exp.arrDim[0] / 2) / exp.pxSize)
        right = raft_loc_x + (0.89 * (exp.arrDim[0] / 2) / exp.pxSize)

        raftCrop = img[int(top):int(bottom), int(left):int(right)]
    return raftCrop