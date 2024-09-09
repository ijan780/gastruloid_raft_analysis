# analyzeFluor_test_GLR.py - Optimized for testing analysis code with single images.

import os
from pathlib import Path
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from skimage import io
from skimage.exposure import rescale_intensity
from skimage.morphology import closing, opening, disk, remove_small_objects, convex_hull_image
from skimage.restoration import rolling_ball
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops, regionprops_table
from skimage.draw import ellipse
from PIL import Image

main_dir = Path(r'C:\Users\MVX\Documents\2 - Data\240729_R2GLR_1.8e6_0-48hrBMP4_0%aneu_3_Analysis\Individual Raft Images')
arr = '1'
raft = '297'
numTP = 9
orderTP = list(range(numTP))
# print(orderTP)

numRafts = 23 * 23
pxSize = 1.0257  # um/px (for 3.2X magnification)
ringWidth = 10 # px
gasRad = math.ceil((500 / 2 / pxSize) / ringWidth) * ringWidth # px rounded to nearest multiple of ringWidth
numRings = int(gasRad / ringWidth)

channels = ['Hull', 'Green']
filenames = []
masks = []

fArea_hull = None
hull_centroid = None

threshVals = [[None] * numTP for i in range(len(channels))]
numObjs = [[None] * numTP for i in range(len(channels))]
fAreas = [[None] * numTP for i in range(len(channels))]
dens = [[None] * numTP for i in range(len(channels))]
eccens = [[None] * numTP for i in range(len(channels))]
radii = [[None] * numTP for i in range(len(channels))]
intMeans = [[None] * numTP for i in range(len(channels))]
intMeans_fold = [None] * numTP
intMeans_perpx = [None] * numTP
intMeans_perpx_fold = [None] * numTP
intMeans_fromCenter = [[None] * int(numRings) for i in range(numTP)]


flag_fail_empty = 0
flag_fail_border = 0
# flag_fail_lo_int = 0

for ch in range(len(channels)):
    if channels[ch] == 'Hull':
        print(f'Now processing: Hull')

        # Import 16 bit image.
        main_dir_hull = Path(r'C:\Users\MVX\Documents\2 - Data\240729_R2GLR_1.8e6_0-48hrBMP4_0%aneu_3_Calcein_Analysis\Individual Raft Images')
        dir = main_dir_hull / f'array{arr}_timepoint00_Violet'
        filename = f'array{arr}_timepoint00_Violet_raft{raft}.tif'
        # dir = main_dir / f'array{arr}_timepoint{orderTP[0]}}_Green'
        # filename = f'array{arr}_timepoint{orderTP[0]:02}_Green_raft{raft}.tif'
        filenames.append(filename)
        image = io.imread(str(dir / filename))
        image_norm = rescale_intensity(image, in_range=(500, 35000))
        # io.imshow(image_norm)
        # plt.show()

        # Create binary mask.
        gaus = gaussian(image, sigma=2, preserve_range=True)
        threshVal = threshold_otsu(gaus)
        thresh = gaus > threshVal
        # io.imshow(thresh)
        # plt.show()
        hull = opening(thresh, disk(5))
        # io.imshow(hull)
        # plt.show()
        hull = remove_small_objects(hull, min_size=25000)
        hull = convex_hull_image(hull)
        # io.imshow(hull)
        # plt.show()
        masks.append(hull)

        fArea_hull = np.sum(hull > 0)
        if (threshVal < 1e4) or (fArea_hull == 0):
            flag_fail_empty = 1
            break

        # Apply convex hull as mask to original thresholded image.
        thresh = np.where(masks[0], thresh, 0).astype(int)

        # Measure hull image.
        labels = label(hull)
        regions = regionprops(labels, image)
        props = regionprops_table(labels, properties=('centroid',
                                                      'orientation',
                                                      'axis_major_length',
                                                      'axis_minor_length',
                                                      'eccentricity',
                                                      'bbox',
                                                      'num_pixels'))
        df_measure = pd.DataFrame(props)
        print(df_measure)

        numObj = df_measure.shape[0]
        nucDen = 1
        df_measure = df_measure.sort_values(by='num_pixels', ascending=False, ignore_index=True)
        eccen = df_measure.loc[0]['eccentricity']
        radius = ((df_measure.loc[0]['axis_major_length'] +
                   df_measure.loc[0]['axis_minor_length']) / 2) / 2
        intMean = np.mean(image[thresh > 0])
        hull_centroid = (df_measure.loc[0]['centroid-0'], df_measure.loc[0]['centroid-1'])
        # print(hull_centroid)

        # Check if bounding box of thresh touches edge of raft.
        bbox_hull = [0, 0, 0, 0]
        rows = np.any(thresh, axis=0)
        cols = np.any(thresh, axis=1)
        bbox_hull[0], bbox_hull[2] = np.where(rows)[0][[0, -1]]
        bbox_hull[1], bbox_hull[3] = np.where(cols)[0][[0, -1]]

        if (bbox_hull[0] < (thresh.shape[0] * 0.01) or bbox_hull[1] < (thresh.shape[1] * 0.01)) or (
                bbox_hull[2] > (thresh.shape[0] * 0.99) or bbox_hull[3] > (thresh.shape[1] * 0.99)):
            print('\tTouching border!')
            flag_fail_border = 1

        threshVals[ch][numTP - 1] = threshVal
        numObjs[ch][numTP - 1] = numObj
        fAreas[ch][numTP - 1] = fArea_hull
        dens[ch][numTP - 1] = nucDen
        eccens[ch][numTP - 1] = eccen
        radii[ch][numTP - 1] = radius
        intMeans[ch][numTP - 1] = intMean

    elif (channels[ch] == 'Green') and (flag_fail_empty == 0):
        print(f'Now processing channel: Green')

        for tp in orderTP:
            # Import 16 bit image.
            dir = main_dir / f'array{arr}_timepoint{tp:02}_{channels[ch]}'
            filename = f'array{arr}_timepoint{tp:02}_{channels[ch]}_raft{raft}.tif'
            filenames.append(filenames)
            image = io.imread(str(dir / filename))
            image_norm = rescale_intensity(image, in_range=(1500, 5000))
            image_crop = image[int(bbox_hull[1]):int(bbox_hull[3]), int(bbox_hull[0]):int(bbox_hull[2])]

            # Create binary mask.
            gaus = gaussian(image, sigma=4, preserve_range=True)
            threshVal1 = threshold_otsu(gaus)
            gaus_crop = gaussian(image_crop, sigma=4, preserve_range=True)
            threshVal2 = threshold_otsu(gaus_crop)
            threshVal3 = threshold_otsu(gaus[masks[0] > 0])

            thresh1 = gaus > threshVal1
            # io.imshow(thresh1)
            # plt.show()
            # thresh2 = gaus > threshVal2
            # io.imshow(thresh2)
            # plt.show()
            # thresh3 = gaus > threshVal3
            # io.imshow(thresh3)
            # plt.show()

            thresh = thresh1

            # Apply convex hull as mask to thresholded image.
            thresh = np.where(masks[0], thresh, 0).astype(int)
            masks.append(thresh)
            fArea = np.sum(thresh > 0)
            # io.imshow(thresh)
            # plt.show()

            # Measure thresholded image.
            labels = label(thresh)
            regions = regionprops(labels, image)
            props = regionprops_table(labels, properties=('centroid',
                                                          'orientation',
                                                          'axis_major_length',
                                                          'axis_minor_length',
                                                          'eccentricity',
                                                          'bbox',
                                                          'num_pixels'))
            df_measure = pd.DataFrame(props)

            # Determine numObj, nucDen, eccen, radius, and intMean.
            numObj = df_measure.shape[0]
            numObjs[ch][tp] = numObj
            if numObj != 0:
                df_measure = df_measure.sort_values(by='num_pixels', ascending=False, ignore_index=True)
                nucDen = fArea / fArea_hull
                eccen = df_measure.loc[0]['eccentricity']
                radius = ((df_measure.loc[0]['axis_major_length'] +
                          df_measure.loc[0]['axis_minor_length']) / 2) / 2
                intMean = np.mean(image[thresh > 0])
                intMean_perpx = intMean / fArea

                threshVals[ch][tp] = threshVal
                fAreas[ch][tp] = fArea
                dens[ch][tp] = nucDen
                eccens[ch][tp] = eccen
                radii[ch][tp] = radius
                intMeans[ch][tp] = intMean
                intMeans_fold[tp] = intMean / intMeans[ch][0]
                intMeans_perpx[tp] = intMean_perpx
                intMeans_perpx_fold[tp] = intMean_perpx / intMeans_perpx[0]

                # Determine intMean from center normalized to intMean via concentric rings at tp = 0.
                if (flag_fail_border == 0) and (intMeans[ch][0] != None):
                    count = 0
                    for r in range(ringWidth, (gasRad + 1), ringWidth):
                        buff = 60 # px
                        half_buff = int(buff / 2) # px

                        image_expand = np.zeros((image.shape[0]+buff, image.shape[1]+buff))
                        image_expand[half_buff:half_buff + image.shape[0], half_buff:half_buff + image.shape[1]] = image
                        hull_expand = np.zeros((image.shape[0]+buff, image.shape[1]+buff))
                        hull_expand[half_buff:half_buff+image.shape[0], half_buff:half_buff+image.shape[1]] = masks[0]

                        ring = np.zeros((image.shape[0] + buff, image.shape[1] + buff))
                        rr, cc = ellipse(hull_centroid[0]+half_buff, hull_centroid[1]+half_buff, r, r)
                        ring[rr, cc] = 1
                        rr_s, cc_s = ellipse(hull_centroid[0]+half_buff, hull_centroid[1]+half_buff,
                                             (r - ringWidth),
                                             (r - ringWidth))
                        ring[rr_s, cc_s] = 0

                        # Only measure pixels in ROI hull.
                        intMean_ring = np.mean(image_expand[(ring > 0) & (hull_expand > 0)])
                        if math.isnan(intMean_ring):
                            break
                        intMeans_fromCenter[tp][count] = intMean_ring / intMeans[ch][0]

                        outPath = Path(r'C:\Users\MVX\Documents\2 - Data\240722_R2GLR_1.8e6_0-48hrBMP4_0%aneu_2_Analysis\From Center')

                        # if tp == orderTP[-1]:
                        #     image *= masks[0]
                        #     plt.imshow(image, cmap='gray')
                        #     ring = ring[half_buff:half_buff+image.shape[0], half_buff:half_buff+image.shape[1]]
                        #     plt.imshow(ring, cmap='jet', alpha=0.5*(ring > 0))
                        #     plt.axis('off')
                        #     file_name = f'Array{arr}_Raft{raft}_TP{tp:02}_FromCenter_{r - 5}.jpg'
                        #     plt.savefig(str(outPath / file_name), bbox_inches='tight', pad_inches=0)

                        count += 1

            # Plot each step of image analysis process.
            fig, ax = plt.subplots(1, 3, figsize=(9, 3), sharey=True)

            ax[0].imshow(image_norm, cmap=plt.cm.gray)
            ax[0].set_title('Norm')

            ax[1].imshow(masks[0], cmap=plt.cm.gray)
            ax[1].imshow(thresh, cmap=plt.cm.jet, alpha=0.9 * (thresh > 0))
            ax[1].set_title('Threshold')

            ax[2].imshow(image_norm, cmap=plt.cm.gray)
            ax[2].set_title('Analysis')

            for props in regions:
                y0, x0 = props.centroid
                orientation = props.orientation
                x1 = x0 + math.cos(orientation) * 0.5 * props.axis_minor_length
                y1 = y0 - math.sin(orientation) * 0.5 * props.axis_minor_length
                x2 = x0 - math.sin(orientation) * 0.5 * props.axis_major_length
                y2 = y0 - math.cos(orientation) * 0.5 * props.axis_major_length

                ax[2].plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
                ax[2].plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
                ax[2].plot(x0, y0, '.g', markersize=15)

                minr, minc, maxr, maxc = props.bbox
                bx = (minc, maxc, maxc, minc, minc)
                by = (minr, minr, maxr, maxr, minr)
                ax[2].plot(bx, by, '-b', linewidth=2.5)

            for a in ax.ravel():
                a.set_axis_off()

            fig.suptitle(filename, fontsize=14)
            fig.tight_layout()

            plt.show()


print()

print(f'Flag_Empty: {flag_fail_empty}')
print(f'Flag_Border: {flag_fail_border}')
# print(f'Flag_Lo_Int: {flag_fail_lo_int}')

print(f'Channels: {channels}')
print(f'Thresh values: {threshVals}')
print(f'Number of objects: {numObjs}')
print(f'Fluor areas: {fAreas}')
print(f'Nuclei densities: {dens}')
print(f'Eccentricity: {eccens}')
print(f'Radius: {radii}')
print(f'Intensity mean: {intMeans}')
print(f'Intensity mean (fold change): {intMeans_fold}')
print(f'Intensity mean per px (fold change): {intMeans_perpx_fold}')

print(f'Intensity mean from center (norm to intMean0): {intMeans_fromCenter}')