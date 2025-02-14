# analyzeFluor_test_Fixed.py - Optimized for testing analysis code with single images.

import os
from pathlib import Path
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from skimage import io, exposure, morphology, segmentation
from skimage.filters import threshold_li
from skimage.measure import label, regionprops, regionprops_table
from skimage.color import label2rgb
from PIL import Image

data_dirs = [Path(r'C:\Users\MVX\Documents\2 - Data\Analysis\240221_RUES2_1.8e6_48hrBMP4_0&100%aneu_2_Analysis'),
             Path(r'C:\Users\MVX\Documents\2 - Data\Analysis\240424_RUES2_1.8e6_48hrBMP4_0&100%aneu_3_Analysis'),
             Path(r'C:\Users\MVX\Documents\2 - Data\Analysis\240501_RUES2_1.8e6_48hrBMP4_0&100%aneu_4_Analysis'),
             Path(r'C:\Users\MVX\Documents\2 - Data\Analysis\240605_RUES2_1.8e6_48hrBMP4_0&100%aneu_5_Analysis'),
             Path(r'C:\Users\MVX\Documents\2 - Data\Analysis\240612_RUES2_1.8e6_48hrBMP4_0&100%aneu_6_Analysis'),
             Path(r'C:\Users\MVX\Documents\2 - Data\Analysis\240619_RUES2_1.8e6_48hrBMP4_0&100%aneu_7_Analysis')
             ]

export_arr = '6'
ploidy = 'An'
raft = '218'

main_dir = data_dirs[int(export_arr) - 1] / 'Individual Raft Images'
if ploidy == 'Eu':
    arr = '1'
elif ploidy == 'An':
    arr = '2'

export_dir = Path(r'C:\Users\MVX\OneDrive - UW\Allbritton Lab\Manuscripts\Manuscript 1\Figures\Sample Eu & Aneu Gastruloids')
export_nucDen = '0.941'

channels = ['Green', 'Blue']
masks = []
numObjs = []

fArea_hull = None
fArea_calc = None
eccen = None
radius = None

nucDen = None
fArea_nuc = None

flag_fail_empty = 0
flag_fail_border = 0

for ch in channels:
    print(f'Now processing channel: {ch}')
    # Import 16 bit image.
    dir = main_dir / f'array{arr}_timepoint00_{ch}'
    filename = f'array{arr}_timepoint00_{ch}_raft{raft}.tif'
    image = io.imread(str(dir / filename))

    # # Crop out raft edges.
    # cropCent = 0.085
    # cropDim = [image.shape[0] * cropCent,
    #            image.shape[0] * (1 - cropCent),
    #            image.shape[1] * cropCent,
    #            image.shape[1] * (1 - cropCent)]
    # cropDim = list(map(int, cropDim))
    # image = image[cropDim[0]:cropDim[1], cropDim[2]:cropDim[3]]

    if ch == 'Green':
        # Rescale intensity.
        image_norm = exposure.rescale_intensity(image, in_range=(1000, 30000))
        # io.imshow(image_norm)
        # plt.show()
        img = Image.fromarray(image_norm)
        # img = img.save(r'C:\Users\MVX\Desktop\Green01.png')

        # Create binary mask.
        # filter = filters.gaussian(image, sigma=4, preserve_range=True)
        thresh_value = threshold_li(image)
        print(thresh_value)

        # Check for empty raft.
        if thresh_value < 2000:
            flag_fail_empty = 1
            numObjs = [None] * len(channels)
            eccen = None
            radius = None
            nucDen = None

            break

        thresh = image > thresh_value
        # io.imshow(thresh)
        # plt.show()
        img = Image.fromarray(thresh)
        # img = img.save(r'C:\Users\MVX\Desktop\Green02.png')

        # Create convex hull surrounding entire gastruloid.
        hull = morphology.closing(thresh, morphology.disk(5))
        # io.imshow(hull)
        # plt.show()

        hull = morphology.remove_small_objects(hull, min_size=25000)
        # io.imshow(hull)
        # plt.show()
        img = Image.fromarray(hull)
        # img = img.save(r'C:\Users\MVX\Desktop\Green03.png')

        # labels = label(hull)
        # hull = segmentation.clear_border(labels)
        # hull = label_clear > 0
        # io.imshow(hull)
        # plt.show()

        hull = morphology.convex_hull_image(hull)
        masks.append(hull)
        io.imshow(hull)
        plt.show
        img = Image.fromarray(hull)
        # img = img.save(r'C:\Users\MVX\Desktop\Green04.png')

        # Check for empty raft.
        fArea_hull = np.sum(hull > 0)
        if fArea_hull == 0:
            flag_fail_empty = 1
            numObjs = [None] * len(channels)
            eccen = None
            radius = None
            nucDen = None

            break

        # Measure hull image.
        labels = label(hull)
        regions = regionprops(labels, image)
        props = regionprops_table(labels, properties=('centroid',
                                                      'orientation',
                                                      'axis_major_length',
                                                      'axis_minor_length',
                                                      'eccentricity',
                                                      'bbox'))
        df_measure = pd.DataFrame(props)

        # Apply convex hull as mask to original thresholded Calcein AM image.
        thresh = np.where(masks[0], thresh, 0).astype(int)
        masks.append(thresh)

        # Determine numObj, nucDen, eccen, and radius.
        numObj = df_measure.shape[0]
        if numObj == 1:
            fArea_calc = np.sum(thresh > 0)
            eccen = df_measure.loc[0]['eccentricity']
            radius = ((df_measure.loc[0]['axis_major_length'] +
                       df_measure.loc[0]['axis_minor_length']) / 2) / 2
        else:
            fArea_calc = None
            eccen = None
            radius = None

        # Check if bounding box of hull touches edge of raft.
        bbox = [0, 0, 0, 0]
        rows = np.any(thresh, axis=0)
        cols = np.any(thresh, axis=1)
        bbox[0], bbox[2] = np.where(rows)[0][[0, -1]]
        bbox[1], bbox[3] = np.where(cols)[0][[0, -1]]
        print(bbox)

        if (bbox[0] < (thresh.shape[0] * 0.01) or bbox[1] < (thresh.shape[1] * 0.01)) or (
                bbox[2] > (thresh.shape[0] * 0.99) or bbox[3] > (thresh.shape[1] * 0.99)):
            flag_fail_border = 1
            print('\tTouching border!')

    elif ch == 'Blue':
        # Rescale intensity.
        image_norm = exposure.rescale_intensity(image, in_range=(1000, 40000))
        # io.imshow(image_norm)
        # plt.show()
        img = Image.fromarray(image_norm)
        img = img.save(export_dir / f'{ploidy}_{export_nucDen}_Arr{export_arr}_{raft:03}_Hoechst.png')
        # img = img.save(r'C:\Users\MVX\Desktop\Blue01.png')

        # Create binary mask.
        # filter = filters.gaussian(image, sigma=2, preserve_range=True)
        thresh = image > 6000
        # io.imshow(thresh)
        # plt.show()
        img = Image.fromarray(thresh)
        # img = img.save(r'C:\Users\MVX\Desktop\Blue02.png')

        #  Apply convex hull as mask to thresholded image.
        thresh = np.where(masks[0], thresh, 0).astype(int)
        masks.append(thresh)
        # io.imshow(thresh)
        # plt.show()
        img = Image.fromarray(thresh > 0)
        img = img.save(export_dir / f'{ploidy}_{export_nucDen}_Arr{export_arr}_{raft:03}_Mask.png')
        # img = img.save(r'C:\Users\MVX\Desktop\Blue03.png')

        # Measure thresholded image.
        labels = label(thresh)
        regions = regionprops(labels, image)
        props = regionprops_table(labels, properties=('centroid',
                                                      'orientation',
                                                      'axis_major_length',
                                                      'axis_minor_length',
                                                      'eccentricity'))
        df_measure = pd.DataFrame(props)

        numObj = df_measure.shape[0]
        fArea_nuc = np.sum(thresh > 0)
        nucDen = fArea_nuc / fArea_hull

    numObjs.append(numObj)

    print(f'\t{df_measure}')

    masks.append(thresh)

    # Plot each step of image analysis process.
    fig, ax = plt.subplots()
    ax.imshow(image_norm, cmap=plt.cm.gray)

    for props in regions:
        y0, x0 = props.centroid
        orientation = props.orientation
        x1 = x0 + math.cos(orientation) * 0.5 * props.axis_minor_length
        y1 = y0 - math.sin(orientation) * 0.5 * props.axis_minor_length
        x2 = x0 - math.sin(orientation) * 0.5 * props.axis_major_length
        y2 = y0 - math.cos(orientation) * 0.5 * props.axis_major_length

        ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
        ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
        ax.plot(x0, y0, '.g', markersize=15)

        minr, minc, maxr, maxc = props.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        ax.plot(bx, by, '-b', linewidth=2.5)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # fig, ax = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
    #
    # ax[0].imshow(image_norm, cmap=plt.cm.gray)
    # ax[0].set_title('Norm')
    #
    # ax[1].imshow(thresh, cmap=plt.cm.gray)
    # ax[1].set_title(f'Threshold')
    #
    # ax[2].imshow(image_norm, cmap=plt.cm.gray)
    # ax[2].set_title('Centroid')
    #
    # for props in regions:
    #     y0, x0 = props.centroid
    #     orientation = props.orientation
    #     x1 = x0 + math.cos(orientation) * 0.5 * props.axis_minor_length
    #     y1 = y0 - math.sin(orientation) * 0.5 * props.axis_minor_length
    #     x2 = x0 - math.sin(orientation) * 0.5 * props.axis_major_length
    #     y2 = y0 - math.cos(orientation) * 0.5 * props.axis_major_length
    #
    #     ax[2].plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
    #     ax[2].plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
    #     ax[2].plot(x0, y0, '.g', markersize=15)
    #
    #     minr, minc, maxr, maxc = props.bbox
    #     bx = (minc, maxc, maxc, minc, minc)
    #     by = (minr, minr, maxr, maxr, minr)
    #     ax[2].plot(bx, by, '-b', linewidth=2.5)
    #
    # for a in ax.ravel():
    #     a.set_axis_off()
    #
    # fig.suptitle(filename, fontsize=14)

    fig.tight_layout()
    plt.show()

print()

print(f'Fail_Empty: {flag_fail_empty}')
print(f'Fail_Border: {flag_fail_border}')

print(f'Channels: {channels}')
print(f'Number of objects: {numObjs}')
print(f'Fluor area (hull): {fArea_hull}')
print(f'Eccentricity: {eccen}')
print(f'Radius: {radius}')
print(f'Fluor area (nuc): {fArea_nuc}')
print(f'Nuclei density: {nucDen}')