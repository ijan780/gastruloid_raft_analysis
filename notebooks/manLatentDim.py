#%%
from pathlib import Path
import os

import pandas as pd
import numpy as np

from skimage import io
from skimage.measure import label, regionprops_table
from skimage.draw import ellipse

import matplotlib.pyplot as plt

#%%
dataDir = Path(r'C:\Users\MVX\OneDrive - UW\Allbritton Lab\Allbritton-Bagheri\Processed')
repPaths = [repPath for repPath in dataDir.iterdir() if repPath.is_dir()]

#%%
# Parameters.
pxSize = 1.6535 * (431/128) # um/px

ringWidth = 25 / pxSize # px
maxRad = (500 / 2) / pxSize # px
buff = 50  # px
half_buff = int(buff / 2)  # px

numTP = 9

#%%
# Intensity-weighted mean radius.
list_reps = []
list_rafts = []
list_tp = []
list_imgNames = []
list_areas = []
list_centroids = []
list_roiRadii = []
list_ringRadii = []
list_intMeans = []
list_intWeightMeanRads = []

# Iterate through replicates.
for repPath in repPaths:
    repName = repPath.name
    os.chdir(repPath)
    print(repName)

    # Iterate through ROIs.
    roiNames = [imgPath.name for imgPath in repPath.iterdir() if "roi" in imgPath.name]
    for roiName in roiNames:

        # Import ROI.
        raft = int(roiName.split("_")[3])
        print(f'\t{roiName}\t{raft}')

        roi_input = io.imread(roiName, as_gray=True)
        roi = (roi_input == 0).astype(np.uint8)  # PNGs save inverted LUT

        # Measure area, centroid, and radius of each ROI.
        area = np.sum(roi) * (pxSize ** 2)
        labels = label(roi)
        props = regionprops_table(labels, properties=('centroid',
                                                      'axis_major_length',
                                                      'axis_minor_length')
                                  )
        df_measure = pd.DataFrame(props)
        roi_centroid = (df_measure.loc[0]['centroid-0'], df_measure.loc[0]['centroid-1'])
        print(f'\t{roi_centroid}')
        roi_radius = ((df_measure.loc[0]['axis_major_length'] +
                       df_measure.loc[0]['axis_minor_length']) / 2) / 2 * pxSize

        # Measure mean intensity 50 um within center and edge of ROI for each time point.
        for tp in range(numTP):

            # Import image.
            imgName = f'{repName}_green_array1_{raft:03}_{tp}.png'
            img = io.imread(imgName, as_gray=True)
            print(f'\t\t{imgName}')

            # Expand images to compensate for gastruloids contacting border.
            img_expand = np.zeros((img.shape[0] + buff, img.shape[1] + buff))
            img_expand[half_buff:half_buff + img.shape[0], half_buff:half_buff + img.shape[1]] = img

            mask = np.zeros((img.shape[0] + buff, img.shape[1] + buff)).astype(int)
            mask[half_buff:half_buff + img.shape[0], half_buff:half_buff + img.shape[1]] = 1

            # # Visualize center/edge masks as overlay.
            # overlay = np.stack([img_expand] * 3, axis=-1).astype(float)
            # color = np.array([255, 0, 0])
            # alpha = 0.4

            # Create and apply circular/ring-shaped mask to processed image.
            ringRadii = []
            intMeans = []

            r = ringWidth
            while r <= maxRad:
                ring = np.zeros((img.shape[0] + buff, img.shape[1] + buff)).astype(int)
                rr, cc = ellipse(roi_centroid[0] + half_buff,
                                 roi_centroid[1] + half_buff,
                                 r,
                                 r)
                ring[rr, cc] = 1
                rr_s, cc_s = ellipse(roi_centroid[0] + half_buff,
                                     roi_centroid[1] + half_buff,
                                     (r - ringWidth),
                                     (r - ringWidth))
                ring[rr_s, cc_s] = 0
                ring = ring * mask

                intMean = np.mean(img_expand[ring > 0])

                ringRadii.append((r - (ringWidth / 2)) * pxSize)
                intMeans.append(intMean)

                # overlay[ring > 0] = ((1 - alpha) * overlay[ring > 0]) + (alpha * color)

                r += ringWidth

            print(f'\t\t\t{ringRadii}')
            print(f'\t\t\t{intMeans}')

            # Calculate radial standard deviation.
            ringRadii = np.array(ringRadii, dtype=float)
            intMeans = np.array(intMeans, dtype=float)

            intWeightMeanRad = np.sum(intMeans * ringRadii) / np.sum(intMeans)  # intensity-weighted mean radius
            intWeightMeanRad_norm = intWeightMeanRad / (maxRad * pxSize)  # dimensionless

            print(f'\t\t\t\t{intWeightMeanRad_norm}')

            # overlay = overlay[half_buff:(img.shape[0] + half_buff), half_buff:(img.shape[1] + half_buff)]
            # overlay = np.clip(overlay, 0, 255).astype(np.uint8)
            # plt.imshow(overlay)
            # plt.title(f'{repName}_green_array1_{raft:03}_{tp}')
            # plt.axis('off')
            # plt.show()

            list_reps.append(repName)
            list_rafts.append(raft)
            list_tp.append(tp)
            list_imgNames.append(imgName)
            list_areas.append(area)
            list_centroids.append(tuple(map(float, roi_centroid)))
            list_roiRadii.append(roi_radius)
            list_ringRadii.append(ringRadii.tolist())
            list_intMeans.append(intMeans.tolist())
            list_intWeightMeanRads.append(intWeightMeanRad_norm)

# Compile data as dataframe.
df_radial = pd.DataFrame({'Rep': list_reps,
                           'Raft': list_rafts,
                           'TimePoint': list_tp,
                           'ImgName': list_imgNames,
                           'Area': list_areas,
                           'Centroid': list_centroids,
                           'roiRadius': list_roiRadii,
                           'RingRadii': list_ringRadii,
                           'IntMeans': list_intMeans,
                           'IntWeightMeanRad': list_intWeightMeanRads
                           })

#%%
# Export df_radGrad.
df_radial.to_csv((dataDir/'radialAnalysis.csv'), index=False)

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
    roiNames = [imgPath.name for imgPath in repPath.iterdir() if "roi" in imgPath.name]
    for roiName in roiNames:

        # Import ROI.
        raft = int(roiName.split("_")[3])
        print(f'\t{roiName}\t{raft}')

        roi_input = io.imread(roiName, as_gray=True)
        roi = (roi_input == 0).astype(np.uint8) # PNGs save inverted LUT

        # Measure area, centroid, and radius of each ROI.
        area = np.sum(roi)
        labels = label(roi)
        props = regionprops_table(labels, properties=('centroid',
                                                      'axis_major_length',
                                                      'axis_minor_length')
                                  )
        df_measure = pd.DataFrame(props)
        centroid = (df_measure.loc[0]['centroid-0'], df_measure.loc[0]['centroid-1'])
        print(f'\t{centroid}')
        radius = ((df_measure.loc[0]['axis_major_length'] +
                   df_measure.loc[0]['axis_minor_length']) / 2) / 2 * pxSize

        # Measure mean intensity 50 um within center and edge of ROI for each time point.
        ringRadii = [ringWidth, radius/pxSize]

        for tp in range(numTP):

            # Import image.
            imgName = f'{repName}_green_array1_{raft:03}_{tp}.png'
            img = io.imread(imgName, as_gray=True)
            print(f'\t\t{imgName}')

            # Expand images to compensate for gastruloids contacting border.
            img_expand = np.zeros((img.shape[0] + buff, img.shape[1] + buff))
            img_expand[half_buff:half_buff + img.shape[0], half_buff:half_buff + img.shape[1]] = img

            mask = np.zeros((img.shape[0] + buff, img.shape[1] + buff)).astype(int)
            mask[half_buff:half_buff + img.shape[0], half_buff:half_buff + img.shape[1]] = 1

            # Visualize center/edge masks as overlay.
            overlay = np.stack([img_expand] * 3, axis=-1).astype(float)
            colors = [np.array([255, 0, 0]), np.array([0, 0, 255])]
            alpha = 0.4

            # Create and apply circular/ring-shaped mask to processed image.
            intMeans = []
            for idx, r in enumerate(ringRadii):
                ring = np.zeros((img.shape[0] + buff, img.shape[1] + buff)).astype(int)
                rr, cc = ellipse(centroid[0] + half_buff,
                                 centroid[1] + half_buff,
                                 r,
                                 r)
                ring[rr, cc] = 1
                rr_s, cc_s = ellipse(centroid[0] + half_buff,
                                     centroid[1] + half_buff,
                                     (r - ringWidth),
                                     (r - ringWidth))
                ring[rr_s, cc_s] = 0
                ring = ring * mask

                intMean = np.mean(img_expand[ring > 0])
                intMeans.append(intMean)

                overlay[ring > 0] = ((1 - alpha) * overlay[ring > 0]) + (alpha * colors[idx])

            print(f'\t\t\t{intMeans}')

            overlay = overlay[half_buff:(img.shape[0] + half_buff), half_buff:(img.shape[1] + half_buff)]
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
            plt.imshow(overlay)
            plt.title(f'{repName}_green_array1_{raft:03}_{tp}')
            plt.axis('off')
            plt.show()

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

#%%
# Export df_radGrad.
df_radGrad.to_csv((dataDir/'radialGradientAnalysis.csv'), index=False)
