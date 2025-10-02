#%%
from pathlib import Path
import os

import pandas as pd
import numpy as np

from skimage import io
from skimage.measure import label, regionprops_table, regionprops
from skimage.draw import ellipse

import matplotlib.pyplot as plt

#%%
dataDir = Path(r'C:\Users\MVX\OneDrive - UW\Allbritton Lab\Allbritton-Bagheri\Processed')
repPaths = [repPath for repPath in dataDir.iterdir()]

#%%
# Parameters.
pxSize = 1.6535 * (128/431) # um/px
ringWidth = 50 / pxSize # px
numTP = 9

#%%
# Radial gradient.
list_reps = []
list_rafts = []
list_tp = []
list_imgNames = []
list_areas = []
list_centroids = []
list_radii = []
list_intMeans_center = []
list_intMeans_edge = []
list_intMeans_grad = []

# Iterate through replicates.
for repPath in repPaths:
    repName = repPath.name
    os.chdir(repPath)
    print(repName)

    # Iterate through ROIs.
    roiNames = [imgPath.name for imgPath in dataDir.iterdir() if "roi" in imgPath.name]
    for roiName in roiNames:
        raft = int(roiName.split("_")[3])
        print(f'\t{roiName}\t{raft}')

        roi = io.imread(roiName)

        # Measure area, centroid, and radius of each ROI.
        area = np.sum(roi)
        labels = label(roi)
        props = regionprops_table(labels, properties=('centroid',
                                                      'axis_major_length',
                                                      'axis_minor_length')
                                  )
        df_measure = pd.DataFrame(props)
        centroid = (df_measure.loc[0]['centroid-0'], df_measure.loc[0]['centroid-1'])
        radius = ((df_measure.loc[0]['axis_major_length'] +
                   df_measure.loc[0]['axis_minor_length']) / 2) / 2 * pxSize

        # Measure mean intensity 50 um within center and edge of ROI for each time point.
        ringRadii = [ringWidth, radius/pxSize]

        for tp in range(numTP):
            imgName = f'{repName}_green_array1_{raft:03}_{tp}.png'
            img = io.imread(imgName)
            print(f'\t\t{imgName}')

            # Create and apply circular/ring-shaped mask to processed image.
            intMeans = []
            for r in ringRadii:
                ring = np.zeros((img.shape[0], img.shape[1]))
                rr, cc = ellipse(centroid[0], centroid[1],
                                 r, r)
                ring[rr, cc] = 1
                rr_s, cc_s = ellipse(centroid[0],
                                     centroid[1],
                                     (r - ringWidth),
                                     (r - ringWidth))
                ring[rr_s, cc_s] = 0

                intMean = np.mean(img[ring > 0])
                intMeans.append(intMean)

            print(f'\t\t\t{intMeans}')

            list_reps.append(repName)
            list_rafts.append(raft)
            list_tp.append(tp)
            list_imgNames.append(imgName)
            list_areas.append(area)
            list_centroids.append(tuple(map(float, centroid)))
            list_radii.append(radius)
            list_intMeans_center.append(intMeans[0])
            list_intMeans_edge.append(intMeans[1])
            list_intMeans_grad.append(intMeans[0] - intMeans[1])

# Compile data as dataframe.
df_radGrad = pd.DataFrame({'Rep': list_reps,
                           'Raft': list_rafts,
                           'TimePoint': list_tp,
                           'ImgName': list_imgNames,
                           'Area': list_areas,
                           'Centroid': list_centroids,
                           'Radius': list_radii,
                           'intMean_Center': list_intMeans_center,
                           'intMean_Edge': list_intMeans_edge,
                           'intMean_Grad': list_intMeans_grad
                           })