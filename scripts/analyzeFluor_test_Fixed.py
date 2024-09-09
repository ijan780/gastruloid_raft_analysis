# analyzeFluor_test_Fixed.py - Optimized for testing analysis code with single images.

import os
from pathlib import Path
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from skimage import io
from skimage.exposure import rescale_intensity
from skimage.morphology import closing, disk, remove_small_objects, convex_hull_image
from skimage.restoration import rolling_ball
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops, regionprops_table
from PIL import Image


main_dir = Path(r'C:\Users\MVX\Documents\2 - Data\231019_RUES2_1.8e6_48hrBMP4_0%aneu_3_Analysis\Individual Raft Images')
arr = '1'
raft = '166'

channels = ['Blue', 'Green', 'Red', 'FarRed']
masks = []
numObjs = []
fArea_hull = None
fAreas = []
nucDens = []
eccens = []
radii = []

flag_fail_empty = 0
flag_fail_border = 0
flag_fail_lo_int = 0

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
    # io.imshow(image)
    # plt.show()

    if ch == 'Blue':
        # Rescale intensity.
        image_norm = rescale_intensity(image, in_range=(500, 20000))
        img = Image.fromarray(image_norm)
        img = img.save(r'C:\Users\MVX\Desktop\Blue01.png')

        # Create binary mask.
        thresh = image > 1500
        # io.imshow(thresh)
        # plt.show()
        img = Image.fromarray(thresh)
        img = img.save(r'C:\Users\MVX\Desktop\Blue02.png')

        # Create convex hull surrounding entire gastruloid.
        hull = closing(thresh, disk(5))
        # io.imshow(hull)
        # plt.show()
        hull = remove_small_objects(hull, min_size=50000)
        # io.imshow(hull)
        # plt.show()
        img = Image.fromarray(hull)
        img = img.save(r'C:\Users\MVX\Desktop\Blue03.png')
        hull = convex_hull_image(hull)
        # io.imshow(hull)
        # plt.show()
        img = Image.fromarray(hull)
        img = img.save(r'C:\Users\MVX\Desktop\Blue04.png')
        masks.append(hull)

        # Check for empty raft.
        fArea_hull = np.sum(hull > 0)
        if fArea_hull == 0:
            flag_fail_empty = 1
            numObjs = [None] * len(channels)
            fAreas = [None] * len(channels)
            nucDens = [None] * len(channels)
            eccen = None
            radius = None
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

        # Apply convex hull as mask to original thresholded Hoechst image and measure fArea.
        thresh = np.where(masks[0], thresh, 0).astype(int)
        masks.append(thresh)
        fArea = np.sum(thresh > 0)

        # Determine numObj, nucDen, eccen, and radius.
        numObj = df_measure.shape[0]
        if numObj == 1:
            nucDen = fArea / fArea_hull
            eccen = df_measure.loc[0]['eccentricity']
            radius = ((df_measure.loc[0]['axis_major_length'] +
                       df_measure.loc[0]['axis_minor_length']) / 2) / 2
        else:
            nucDen = None
            eccen = None
            radius = None

        # Check if bounding box of thresh touches edge of raft.
        bbox = [0, 0, 0, 0]
        rows = np.any(thresh, axis=0)
        cols = np.any(thresh, axis=1)
        bbox[0], bbox[2] = np.where(rows)[0][[0, -1]]
        bbox[1], bbox[3] = np.where(cols)[0][[0, -1]]

        if (bbox[0] < (thresh.shape[0] * 0.01) or bbox[1] < (thresh.shape[1] * 0.01)) or (
                bbox[2] > (thresh.shape[0] * 0.99) or bbox[3] > (thresh.shape[1] * 0.99)):
            flag_fail_border = 1
            print('\tTouching border!')

    else:
        if ch == 'Green':
            # Rescale intensity.
            image_norm = rescale_intensity(image, in_range=(200, 1500))

        elif ch == 'Red':
            # Rescale intensity & processing.
            image_norm = rescale_intensity(image, in_range=(200, 15000))
            image = image - rolling_ball(image, radius=50)
            image = gaussian(image, sigma=2, preserve_range=True)

        elif ch == 'FarRed':
            # Rescale intensity & processing.
            image_norm = rescale_intensity(image, in_range=(200, 1500))
            image = image - rolling_ball(image, radius=50)
            image = gaussian(image, sigma=2, preserve_range=True)

        img = Image.fromarray(image_norm)
        img_out = Path(r'C:\Users\MVX\Desktop')
        img = img.save(str(img_out / f'{ch}01.png'))

        # Create binary mask.
        thresh = image > threshold_otsu(image)
        lo_int = [((ch == 'Green') and (threshold_otsu(image) < 200)),
                  ((ch == 'Red') and (threshold_otsu(image) < 300)),
                  ((ch == 'FarRed') and (threshold_otsu(image) < 200))
                  ]
        if True in lo_int:
            flag_fail_lo_int = 1
        print(f'{ch} Thresh: {threshold_otsu(image)}')
        # io.imshow(thresh)
        # plt.show()
        img = Image.fromarray(thresh)
        img_out = Path(r'C:\Users\MVX\Desktop')
        img = img.save(str(img_out / f'{ch}02.png'))

        # Apply complete hull to other channels.
        thresh = np.where(masks[0], thresh, 0).astype(int)
        masks.append(thresh)
        img = Image.fromarray(thresh > 0)
        img_out = Path(r'C:\Users\MVX\Desktop')
        img = img.save(str(img_out / f'{ch}03.png'))

        labels = label(thresh)
        regions = regionprops(labels, image)
        props = regionprops_table(labels, properties=('centroid',
                                                      'orientation',
                                                      'axis_major_length',
                                                      'axis_minor_length',
                                                      'eccentricity',
                                                      'num_pixels'))
        df_measure = pd.DataFrame(props)

        numObj = df_measure.shape[0]
        fArea = np.sum(thresh > 0)
        nucDen = fArea / fArea_hull
        df_measure = df_measure.sort_values(by='num_pixels', ascending=False, ignore_index=True)
        if numObj > 0:
            eccen = df_measure.loc[0]['eccentricity']
            radius = ((df_measure.loc[0]['axis_major_length'] +
                       df_measure.loc[0]['axis_minor_length']) / 2) / 2
        else:
            eccen = None
            radius = None

        # # Only measure eccen and radius if only single, large object.
        # df_measure = df_measure.sort_values(by='num_pixels', ascending=False, ignore_index=True)
        # if (numObj == 1) or (numObj > 0 and df_measure.loc[0]['num_pixels'] / df_measure.loc[1]['num_pixels'] > 5):
        #     eccen = df_measure.loc[0]['eccentricity']
        #     radius = (df_measure.loc[0]['axis_major_length'] +
        #               df_measure.loc[0]['axis_minor_length']) / 2
        # else:
        #     eccen = None
        #     radius = None

    numObjs.append(numObj)
    fAreas.append(fArea)
    nucDens.append(nucDen)
    eccens.append(eccen)
    radii.append(radius)

    print(f'\t{df_measure}')

    # Plot each step of image analysis process.
    fig, ax = plt.subplots(1, 3, figsize=(9, 3), sharey=True)

    ax[0].imshow(image_norm, cmap=plt.cm.gray)
    ax[0].set_title('Norm')

    ax[1].imshow(thresh, cmap=plt.cm.gray)
    ax[1].set_title('Threshold')

    ax[2].imshow(image_norm, cmap=plt.cm.gray)
    ax[2].set_title('Centroid')

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
print(f'Flag_Lo_Int: {flag_fail_lo_int}')

print(f'Channels: {channels}')
print(f'Fluor area (hull): {fArea_hull}')
print(f'Number of objects: {numObjs}')
print(f'Fluor areas: {fAreas}')
print(f'Nuclei densities: {nucDens}')
print(f'Eccentricity: {eccens}')
print(f'Radius: {radii}')