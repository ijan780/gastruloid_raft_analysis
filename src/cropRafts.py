# cropRafts.py - Locates and crops individual rafts for image analysis.

from pathlib import Path
from scipy.io import loadmat

from skimage import io
import numpy as np
import tifffile as tiff

def cropRafts(exp):
    # Initialize numRafts, channels, and mask time point.
    numRafts = exp.arrDim[2] ** 2
    if exp.dataDir == exp.dataDir:
        ch_end = exp.channels_end
    else:
        ch_end = [exp.channel_mask]
    ch_tl = exp.channels_tl

    endDirSame = (exp.dataDir == exp.endDir)
    if endDirSame:
        endTP = exp.numTimePoints - 1
    else:
        endTP = 0

    # Load Loc_cell.mat (links local/global indices) and Centroid_cell.mat (raft px location within subsection) for endpoint.
    loc_cell2_end = loadmat(exp.endDir / 'data' / 'Loc_cell2.mat')
    loc_cell2_end = loc_cell2_end['Loc_cell2']
    centroid_cell_end = loadmat(exp.endDir / 'data' / 'Centroid_cell.mat')
    centroid_cell_end = centroid_cell_end['Centroid_cell']

    # Load Loc_cell.mat (links local/global indices) and Centroid_cell.mat (raft px location within subsection) for data.
    loc_cell2 = loadmat(exp.dataDir / 'data' / 'Loc_cell2.mat')
    loc_cell2 = loc_cell2['Loc_cell2']
    centroid_cell = loadmat(exp.dataDir / 'data' / 'Centroid_cell.mat')
    centroid_cell = centroid_cell['Centroid_cell']

    # Set up main output folder for individual cropped raft images.
    outDir_main = Path(f'{exp.dataDir}_Analysis') / 'Individual Raft Images'
    outDir_main.mkdir(exist_ok=True)
    outDir_end = Path(outDir_main) / 'Endpoint'
    outDir_end.mkdir(exist_ok=True)
    outDir_tl = Path(outDir_main) / 'Time-lapse'
    outDir_tl.mkdir(exist_ok=True)

    # Loop through loc_cell for each raft and locate corresponding array subsection folder.
    for arr in range(exp.numArrs):
        raftCount = 0
        for raft in range(0, numRafts):
            # Check if raft exists at every time point (including separate mask imaging) and if loc_idx is within num of rafts in centroid_cell.
            if endDirSame:
                raft_loc_cell2 = [tp[raft] for tp in loc_cell2[arr, :]]
            else:
                raft_loc_cell2 = [tp[raft] for tp in loc_cell2[arr, :]] + [tp[raft] for tp in loc_cell2_end[arr, :]]

            raft_centroid_cell_count = []
            for tp in range(exp.numTimePoints):
                # Identify imaging row/col and local index.
                img_row = loc_cell2[arr, tp][raft, 1] - 1
                img_col = loc_cell2[arr, tp][raft, 2] - 1
                raft_centroid_cell_count.append(centroid_cell[arr, tp][img_row, img_col].shape[0])
            if not endDirSame:
                # Identify imaging row/col and local index of endpoint images.
                img_row_end = loc_cell2_end[arr, endTP][raft, 1] - 1
                img_col_end = loc_cell2_end[arr, endTP][raft, 2] - 1
                raft_centroid_cell_count.append(centroid_cell_end[arr, endTP][img_row_end, img_col_end].shape[0])
            raft_loc_idx = [tp[3] for tp in raft_loc_cell2] # list of local indices for time-series
            # print(f'\t{raft_loc_idx}')
            # print(f'\t{raft_centroid_cell_count}')
            raft_coord_exist = [a <= b for a, b in zip(raft_loc_idx, raft_centroid_cell_count)] # a <= b because MATLAB starts at 1
            # print(f'\t{False in raft_coord_exist}')

            if (0 in raft_loc_idx) or (False in raft_coord_exist):
                continue
            raftCount += 1
            # print(f'\t Array {arr + 1} Raft {raft + 1}')

            # Crop raft at final time point for all endpoint channels then stack into single TIFF as (c, y, x).
            crop_end = []
            # Identify imaging row/col and local index.
            img_row_end = loc_cell2_end[arr, endTP][raft, 1] - 1
            img_col_end = loc_cell2_end[arr, endTP][raft, 2] - 1
            loc_idx_end = loc_cell2_end[arr, endTP][raft, 3] - 1
            for ch in ch_end:
                raftCrop = crop(exp, exp.endDir,
                                img_row_end, img_col_end, loc_idx_end,
                                centroid_cell_end,
                                arr, endTP, ch)
                crop_end.append(raftCrop)
            end_img = np.stack(crop_end, axis=0)
            outDir_end_raft = outDir_end / f'array{arr + 1}_raft{(raft + 1):03}_end.tif'
            tiff.imwrite(outDir_end_raft, end_img,
                         imagej=True, metadata={'axes': 'CYX', 'channels': ch_end})

            # Crop raft at each time point for all time-lapse channels then stack into single TIFF as (tp, c, y, x).
            crop_tp = []
            for tp in range(exp.numTimePoints):
                # Identify imaging row/col and local index.
                img_row = loc_cell2[arr, tp][raft, 1] - 1
                img_col = loc_cell2[arr, tp][raft, 2] - 1
                loc_idx = loc_cell2[arr, tp][raft, 3] - 1

                crop_ch = []
                for ch in ch_tl:
                    raftCrop = crop(exp, exp.dataDir,
                                    img_row, img_col, loc_idx,
                                    centroid_cell,
                                    arr, tp, ch)
                    crop_ch.append(raftCrop)
                stacked_ch = np.stack(crop_ch, axis=0)
                crop_tp.append(stacked_ch)
            tl_img = np.stack(crop_tp, axis=0)
            outDir_tl_raft = outDir_tl / f'array{arr + 1}_raft{(raft + 1):03}_tl.tif'
            tiff.imwrite(outDir_tl_raft, tl_img,
                         imagej=True, metadata={'axes': 'TCYX', 'channels': ch_tl})

        print(f'\t(Array {arr + 1}) Cropped {raftCount} rafts')

    print('Crop complete.')


# Loop through loc_cell for each raft and locate corresponding array subsection folder.
def crop(exp, dir, img_row, img_col, loc_idx, centroid_cell, arr, tp, ch):
    # Identify local px coordinate for specific raft.
    raft_loc_x = centroid_cell[arr, tp][img_row, img_col][loc_idx][0]
    raft_loc_y = centroid_cell[arr, tp][img_row, img_col][loc_idx][1]

    # Locate subsection folder and specific image file for Time Point 'tp' and Channel 'ch'.
    imgDir = dir / f'{arr + 1}' / f'{img_row + 1}x{img_col + 1}' / f'{tp:02}_{ch}.tif'

    with tiff.TiffFile(imgDir) as tif:
        img = tif.asarray()

    # Crop image to specific raft.
    top = raft_loc_y - (0.89 * (exp.arrDim[0] / 2) / exp.pxSize)
    bottom = raft_loc_y + (0.89 * (exp.arrDim[0] / 2) / exp.pxSize)
    left = raft_loc_x - (0.89 * (exp.arrDim[0] / 2) / exp.pxSize)
    right = raft_loc_x + (0.89 * (exp.arrDim[0] / 2) / exp.pxSize)

    raftCrop = img[int(top):int(bottom), int(left):int(right)]

    return raftCrop