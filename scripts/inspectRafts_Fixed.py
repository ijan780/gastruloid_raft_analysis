# inspectRafts_CalcHoechst.py - Standalone program that loops through all pass and failed rafts from analyzeFluor_Fixed.py

import os
from pathlib import Path
import pandas as pd
import numpy as np
import math

from skimage import io
from skimage.exposure import rescale_intensity
from skimage.morphology import closing, disk, remove_small_objects, convex_hull_image
from skimage.restoration import rolling_ball
from skimage.filters import gaussian, threshold_otsu
import matplotlib.pyplot as plt


# Specify data path for analysis folder.
analPath = Path(r'C:\Users\MVX\Documents\2 - Data\231019_RUES2_1.8e6_48hrBMP4_0%aneu_3_Analysis')
arr = 1
anal_type = 'Fixed'

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
fdataPath = Path(analPath / f'array{arr}_fluorAnalysis_{anal_type}.csv')
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

            elif ch == 'Blue':
                image_norm = rescale_intensity(image, in_range=(500, 20000))
                thresh = image > 1500
                hull = closing(thresh, disk(5))
                hull = remove_small_objects(hull, min_size=25000)
                hull = convex_hull_image(hull)
                masks.append(hull)
                thresh = np.where(hull, thresh, 0).astype(int)
                masks.append(thresh)

            else:
                if ch == 'Green':
                    image_norm = rescale_intensity(image, in_range=(200, 1500))

                elif ch == 'Red':
                    image_norm = rescale_intensity(image, in_range=(200, 15000))
                    image = image - rolling_ball(image, radius=50)
                    image = gaussian(image, sigma=2, preserve_range=True)

                elif ch == 'FarRed':
                    image_norm = rescale_intensity(image, in_range=(200, 1500))
                    image = image - rolling_ball(image, radius=50)
                    image = gaussian(image, sigma=2, preserve_range=True)

                thresh = image > threshold_otsu(image)
                thresh = np.where(hull, thresh, 0).astype(int)
                masks.append(thresh)

            images_norm.append(image_norm)

        # Plot figure with img_norm and mask for each channel.
        fig, ax = plt.subplots(2, 5, figsize=(6, 3), sharey=True)

        ax[0, 0].imshow(images_norm[0], cmap=plt.cm.gray)
        ax[0, 0].set_title('BF')

        ax[0, 1].imshow(images_norm[1], cmap=plt.cm.gray)
        ax[0, 1].set_title('Blue')
        ax[1, 1].imshow(masks[0], cmap=plt.cm.gray)
        ax[1, 1].imshow(masks[1], cmap=plt.cm.jet, alpha=0.9 * (masks[1] > 0))

        ax[0, 2].imshow(images_norm[2], cmap=plt.cm.gray)
        ax[0, 2].set_title('Green')
        ax[1, 2].imshow(masks[0], cmap=plt.cm.gray)
        ax[1, 2].imshow(masks[2], cmap=plt.cm.jet, alpha=0.9 * (masks[2] > 0))

        ax[0, 3].imshow(images_norm[3], cmap=plt.cm.gray)
        ax[0, 3].set_title('Red')
        ax[1, 3].imshow(masks[0], cmap=plt.cm.gray)
        ax[1, 3].imshow(masks[3], cmap=plt.cm.jet, alpha=0.9 * (masks[3] > 0))

        ax[0, 4].imshow(images_norm[4], cmap=plt.cm.gray)
        ax[0, 4].set_title('FarRed')
        ax[1, 4].imshow(masks[0], cmap=plt.cm.gray)
        ax[1, 4].imshow(masks[4], cmap=plt.cm.jet, alpha=0.9 * (masks[4] > 0))

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