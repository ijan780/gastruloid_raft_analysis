# cropRafts.py - Locates and crops individual rafts for image analysis.

from pathlib import Path
from scipy.io import loadmat
from tqdm import tqdm

from skimage import io
import numpy as np
import tifffile as tiff

def cropRafts(exp):
    # Initialize numRafts, channels, and mask time point.
    endDirSame = (exp.dataDir == exp.endDir)

    numRafts = exp.arrDim[2] ** 2

    ch_end = exp.channels_end

    ch_tl = exp.channels_tl

    rep_and_stain = exp.reporters_and_stains

    if endDirSame:
        endTP = exp.numTimePoints - 1
    else:
        endTP = 0

    if not endDirSame:
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
        skipRafts = {}
        pbar = tqdm(range(0, numRafts), desc=f'\tArray {arr + 1}')
        for raft in pbar:
            # Check if raft exists at every time point (including separate mask imaging) and if locIdx is within num of rafts in centroid_cell.
            raft_centroid_cell_count = []
            for tp in range(exp.numTimePoints):
                img_row, img_col, _ = getLocIdx(loc_cell2, arr, tp, raft)
                raft_centroid_cell_count.append(centroid_cell[arr, tp][img_row, img_col].shape[0])

            if endDirSame:
                raft_loc_cell2 = [tp[raft] for tp in loc_cell2[arr, :]]
            else:
                img_row_end, img_col_end, _ = getLocIdx(loc_cell2_end, arr, endTP, raft)
                raft_centroid_cell_count.append(centroid_cell_end[arr, endTP][img_row_end, img_col_end].shape[0])

                raft_loc_cell2 = [tp[raft] for tp in loc_cell2[arr, :]] + [tp[raft] for tp in loc_cell2_end[arr, :]]

            raft_locIdx = [tp[3] for tp in raft_loc_cell2] # list of local indices for tl
            raft_coord_exist = [a <= b for a, b in zip(raft_locIdx, raft_centroid_cell_count)] # a <= b because MATLAB starts at 1

            # Skip rafts that DNE at any time point.
            if (0 in raft_locIdx) or (False in raft_coord_exist):
                continue

            # # TODO: Skip rafts if and only if DNE across entire time-series.
            # if all(x == 0 for x in raft_locIdx) or not any(raft_coord_exist):
            #     continue
            # elif (0 in raft_locIdx) or (False in raft_coord_exist):
            #     img_row_med = np.median([tp[0] for tp in raft_loc_cell2])
            #     img_col_med = np.median([tp[1] for tp in raft_loc_cell2])
            #     raft_locIdx_med = np.median(raft_locIdx)
            #     tp_med = raft_locIdx.index(raft_locIdx_med)
            #     skipRafts[raft] = [img_row_med, img_col_med, raft_locIdx_med, tp_med]

            raftCount += 1 # count only rafts that exist
            # print(f'\t Array {arr + 1} Raft {raft + 1}')

            # Crop raft at final time point for all endpoint channels then stack into single TIFF as (c, y, x).
            if ch_end:
                crop_end = []
                # TODO: if raft is in missing dict then get img_row/col + locIdx from dict, else:
                img_row_end, img_col_end, locIdx_end = getLocIdx(loc_cell2_end, arr, endTP, raft)

                for ch in ch_end:
                    raftCrop = crop(exp, exp.endDir,
                                    img_row_end, img_col_end, locIdx_end,
                                    centroid_cell_end,
                                    arr, endTP, ch)
                    crop_end.append(raftCrop)
                end_img = np.stack(crop_end, axis=0)
                outPath_end_raft = outDir_end / f'array{arr + 1}_raft{(raft + 1):03}_end.tif'
                tiff.imwrite(outPath_end_raft, end_img,
                             imagej=True, metadata={'axes': 'CYX',
                                                    'channels': [rep_and_stain[f'end_{ch}'] for ch in ch_end]})

            # Crop raft at each time point for all time-lapse channels then stack into single TIFF as (tp, c, y, x).
            if exp.numTimePoints > 1:
                crop_tp = []
                for tp in range(exp.numTimePoints):
                    # TODO: if raft is in missing dict then get img_row/col + locIdx from dict, else:
                    img_row, img_col, locIdx = getLocIdx(loc_cell2, arr, tp, raft)
                    crop_ch = []
                    for ch in ch_tl:
                        raftCrop = crop(exp, exp.dataDir,
                                        img_row, img_col, locIdx,
                                        centroid_cell,
                                        arr, tp, ch)
                        crop_ch.append(raftCrop)
                    stacked_ch = np.stack(crop_ch, axis=0)
                    crop_tp.append(stacked_ch)
                tl_img = np.stack(crop_tp, axis=0)
                outPath_tl_raft = outDir_tl / f'array{arr + 1}_raft{(raft + 1):03}_tl.tif'
                tiff.imwrite(outPath_tl_raft, tl_img,
                             imagej=True, metadata={'axes': 'TCYX',
                                                    'channels': [rep_and_stain[f'tl_{ch}'] for ch in ch_tl]})

        pbar.close()
        print(f'\t(Array {arr + 1}) Cropped {raftCount} rafts')

    print('Crop complete.')


# Get img_row, img_col, and locIdx from loc_cell2.
def getLocIdx(loc_cell2, arr, tp, raft):
    img_row = loc_cell2[arr, tp][raft, 1] - 1
    img_col = loc_cell2[arr, tp][raft, 2] - 1
    locIdx = loc_cell2[arr, tp][raft, 3] - 1
    return img_row, img_col, locIdx


# Get local px coordinates for specific raft.
def getCentroid(img_row, img_col, locIdx, centroid_cell, arr, tp):
    raft_loc_x = centroid_cell[arr, tp][img_row, img_col][locIdx][0]
    raft_loc_y = centroid_cell[arr, tp][img_row, img_col][locIdx][1]
    return raft_loc_x, raft_loc_y


# Loop through loc_cell for each raft and locate corresponding array subsection folder.
def crop(exp, dir, img_row, img_col, locIdx, centroid_cell, arr, tp, ch, tp_cen=None):
    if tp_cen is None:
        tp_cen = tp

    # Find local px coordinates.
    raft_loc_x, raft_loc_y = getCentroid(img_row, img_col, locIdx, centroid_cell, arr, tp_cen)

    # Locate subsection folder and specific image file for Time Point 'tp' and Channel 'ch'.
    imgPath = dir / f'{arr + 1}' / f'{img_row + 1}x{img_col + 1}' / f'{tp:02}_{ch}.tif'

    with tiff.TiffFile(imgPath) as tif:
        img = tif.asarray()

    # Crop image to specific raft.
    top = raft_loc_y - (0.89 * (exp.arrDim[0] / 2) / exp.pxSize)
    bottom = raft_loc_y + (0.89 * (exp.arrDim[0] / 2) / exp.pxSize)
    left = raft_loc_x - (0.89 * (exp.arrDim[0] / 2) / exp.pxSize)
    right = raft_loc_x + (0.89 * (exp.arrDim[0] / 2) / exp.pxSize)

    raftCrop = img[int(top):int(bottom), int(left):int(right)]

    return raftCrop