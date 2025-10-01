#%%
from pathlib import Path
import os
import math
from tqdm import tqdm

import tifffile as tiff
import pandas as pd
import numpy as np
from skimage import io
from skimage.measure import label, regionprops_table, regionprops
from skimage.draw import ellipse
import matplotlib.pyplot as plt

#%%
dataDir = Path(r'C:\Users\MVX\OneDrive - UW\Allbritton Lab\Allbritton-Bagheri\251001_Manual Analysis Test\250609')
os.chdir(dataDir)

#%%
imgNames = [imgPath.name for imgPath in dataDir.iterdir() if "roi" not in imgPath.name]

#%%
# Parameters.
pxSize = 1.6535 # um/px
ringWidth = 50 / pxSize # px
numTP = 9

#%%
# Radial gradient.
list_imgNames =[]
list_tp = []
list_intMeans_center = []
list_intMeans_edge = []
list_intMeans_grad = []

for imgName in imgNames:
    # Measure ROI area & centroid.
    roiName = f'{imgName[:-4]}_roi.tif'
    with tiff.TiffFile(roiName) as tif:
        roi = tif.asarray()
    labels = label(roi)
    props = regionprops_table(labels, properties=('centroid',
                                                  'axis_major_length',
                                                  'axis_minor_length')
                              )
    df_measure = pd.DataFrame(props)
    fArea = np.sum(roi)
    centroid = (df_measure.loc[0]['centroid-0'], df_measure.loc[0]['centroid-1'])
    radius = ((df_measure.loc[0]['axis_major_length'] +
               df_measure.loc[0]['axis_minor_length']) / 2) / 2 * pxSize

    # Measure mean intensity 50 um within center and edge of ROI.
    radii = [ringWidth, radius/pxSize]

    with tiff.TiffFile(imgName) as tif:
        img = tif.asarray()

    sox = img[:, 1, :, :]

    for tp in range(numTP):
        slice = sox[tp, :, :]
        intMeans = []

        # Create circular/ring-shaped mask
        for r in radii:
            ring = np.zeros((slice.shape[0], slice.shape[1]))
            rr, cc = ellipse(centroid[0], centroid[1],
                             r, r)
            ring[rr, cc] = 1
            rr_s, cc_s = ellipse(centroid[0],
                                 centroid[1],
                                 (r - ringWidth),
                                 (r - ringWidth))
            ring[rr_s, cc_s] = 0

            intMean = np.mean(slice[ring > 0])
            intMeans.append(intMean)

        list_imgNames.append(imgName)
        list_tp.append(tp)
        list_intMeans_center.append(intMeans[0])
        list_intMeans_edge.append(intMeans[1])
        list_intMeans_grad.append(intMeans[0] - intMeans[1])

df_radGrad = pd.DataFrame({'imgName': list_imgNames,
                           'timePoint': list_tp,
                           'intMean_Center': list_intMeans_center,
                           'intMean_Edge': list_intMeans_edge,
                           'intMean_Grad': list_intMeans_grad
                           })