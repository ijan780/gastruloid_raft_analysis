# inspectRafts_CalcHoechst.py - Standalone program that loops through all pass and failed rafts from analyzeFluor_Fixed.py

import os
from pathlib import Path
import pandas as pd
import numpy as np
import math

from skimage import io
from skimage.exposure import rescale_intensity
from skimage.morphology import closing, disk, remove_small_objects, convex_hull_image
from skimage.filters import threshold_li
import matplotlib.pyplot as plt


# Specify data path for analysis folder.
analPath = Path(r'C:\Users\MVX\Documents\2 - Data\Analysis\240619_RUES2_1.8e6_48hrBMP4_0&100%aneu_7_Analysis')
arr = 2
anal_type = 'CalcHoechst'

# Determine paths for list of graded rafts.
gradedPath = Path(analPath / f'array{arr}_raftGrading_{anal_type}.csv')
df_grad = pd.read_csv(str(gradedPath))

# Initialize both pass/fail raft indices.
raftGrad_cat = ['Pass', 'Fail_eccen', 'Excl_empty', 'Excl_border']
raftGrads = [None] * len(raftGrad_cat)
for i in range(len(raftGrad_cat)):
    raftList = list(df_grad[raftGrad_cat[i]])
    raftGrads[i] = [int(f) for f in raftList if not math.isnan(f)]

# print(raftGrads)

# Import fluor data frame.
fdataPath = Path(analPath / f'array{arr}_fluorAnalysis_{anal_type}_Raw.csv')
df_fdata = pd.read_csv(str(fdataPath))

# Truncate df_fdata to final time point.
metaPath = Path(analPath / 'Experiment Metadata.csv')
df_meta = pd.read_csv(str(metaPath))
interval = float(df_meta['Values'].values[1])
tp = int(float(df_meta['Values'].values[3]) - 1)
df_final = df_fdata[df_fdata['Time'] == interval * tp]

# Main channels
channels = []
if anal_type == 'CalcHoechst':
    channels = ['BF', 'Green', 'Blue']
elif anal_type == 'Fixed':
    channels = ['BF', 'Blue', 'Green', 'Red', 'FarRed']

# Create save directories.
outPath_pass = Path(f'{df_meta.iat[0, 1]}_Analysis') / f'array{arr}_Rafts Pass ({anal_type})'
os.makedirs(outPath_pass, exist_ok=True)
outPath_fail = Path(f'{df_meta.iat[0, 1]}_Analysis') / f'array{arr}_Rafts Fail ({anal_type})'
os.makedirs(outPath_fail, exist_ok=True)
outPath_excl = Path(f'{df_meta.iat[0, 1]}_Analysis') / f'array{arr}_Rafts Excl ({anal_type})'
os.makedirs(outPath_excl, exist_ok=True)

# Inspect each raft.
for i in range(len(raftGrads)):
    for raft in raftGrads[i]:
        images_norm = []
        masks = []

        for ch in channels:
            imgPath = Path(analPath /
                           'Individual Raft Images' /
                           f'array{arr}_timepoint{tp:02}_{ch}' /
                           f'array{arr}_timepoint{tp:02}_{ch}_raft{int(raft):03}.tif')
            image = io.imread(str(imgPath))

            if ch == 'BF':
                image_norm = rescale_intensity(image)

            elif ch == 'Green':
                image_norm = rescale_intensity(image, in_range=(1000, 65000))
                thresh_value = threshold_li(image)
                thresh = image > thresh_value
                hull = closing(thresh, disk(5))
                hull = remove_small_objects(hull, min_size=25000)
                hull = convex_hull_image(hull)
                masks.append(hull)
                thresh = np.where(hull, thresh, 0).astype(int)
                masks.append(thresh)

            elif ch == 'Blue':
                image_norm = rescale_intensity(image, in_range=(1000, 40000))
                thresh = image > 6000
                thresh = np.where(masks[0], thresh, 0).astype(int)
                masks.append(thresh)

            images_norm.append(image_norm)

        # Plot figure with img_norm and mask for each channel.
        fig, ax = plt.subplots(2, 3, figsize=(5, 4), sharey=True)

        ax[0, 0].imshow(images_norm[0], cmap=plt.cm.gray)
        ax[0, 0].set_title('BF')

        ax[0, 1].imshow(images_norm[1], cmap=plt.cm.gray)
        ax[0, 1].set_title('Green')
        ax[1, 1].imshow(masks[0], cmap=plt.cm.gray)
        ax[1, 1].imshow(masks[1], cmap=plt.cm.jet, alpha=0.9 * (masks[1] > 0))

        ax[0, 2].imshow(images_norm[2], cmap=plt.cm.gray)
        ax[0, 2].set_title('Blue')
        ax[1, 2].imshow(masks[0], cmap=plt.cm.gray)
        ax[1, 2].imshow(masks[2], cmap=plt.cm.jet, alpha=0.9 * (masks[2] > 0))

        for a in ax.ravel():
            a.set_axis_off()

        fig.suptitle(f'Raft {int(raft):03} ({raftGrad_cat[i]})', fontsize=14)
        fig.tight_layout()

        if i == 0:
            plt.savefig(str(Path(outPath_pass / f'array{arr}_{raftGrad_cat[i]}_raft{int(raft):03}.png')))
        elif i == 1:
            plt.savefig(str(Path(outPath_fail / f'array{arr}_{raftGrad_cat[i]}_raft{int(raft):03}.png')))
        else:
            plt.savefig(str(Path(outPath_excl / f'array{arr}_{raftGrad_cat[i]}_raft{int(raft):03}.png')))

        plt.close()
        # plt.show()