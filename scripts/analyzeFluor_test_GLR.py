# analyzeFluor_test_GLR.py - Optimized for testing analysis code with single images.

from pathlib import Path
import tifffile as tiff
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from skimage import io
from skimage.exposure import rescale_intensity

from src.analyzeFluor_GLR import segment_hull, segment_germ, measure_genFeat, ROI_borderCheck, fromCenter

main_dir = Path(r'C:\Users\MVX\Documents\2 - Data\240805_R2GLR_1.8e6_0-48hrBMP4_0%aneu_4_Analysis\Individual Raft Images')

arr = '1'
raft = '207'
numTP = 9

numRafts = 23 * 23
pxSize = 1.0257  # um/px (for 3.2X magnification)
ringWidth = 10 # px
gasRad = math.ceil((500 / 2 / pxSize) / ringWidth) * ringWidth # px rounded to nearest multiple of ringWidth
numRings = int(gasRad / ringWidth)

channels = ['Hull', 'Green']
masks = []

fArea_hull = None
hull_centroid = None

threshVals = [[None] * numTP for _ in range(len(channels))]
numObjs = [[None] * numTP for _ in range(len(channels))]
fAreas = [[None] * numTP for _ in range(len(channels))]
dens = [[None] * numTP for _ in range(len(channels))]
eccens = [[None] * numTP for _ in range(len(channels))]
radii = [[None] * numTP for _ in range(len(channels))]
intMeans = [[None] * numTP for _ in range(len(channels))]
intMeans_fold = [None] * numTP
intMeans_perpx = [None] * numTP
intMeans_perpx_fold = [None] * numTP
intMeans_fromCenter = [[None] * int(numRings) for _ in range(numTP)]
intMeans_radialSlope = [None] * numTP

flag_fail_empty = 0
flag_fail_border = 0

for idx, ch in enumerate(channels):
    if channels[idx] == 'Hull':
        print(f'Now processing: Hull')

        # Import 16 bit image.
        hullDir = main_dir / 'Endpoint' / f'array{arr}_raft{raft}_end.tif'
        with tiff.TiffFile(hullDir) as tif:
            image = tif.pages[0].asarray()
        image_norm = rescale_intensity(image, in_range=(500, 35000))
        # io.imshow(image_norm)
        # plt.show()

        # Create binary mask.
        hull, thresh, threshVal = segment_hull(image, flag_show=False)
        masks.append(hull)

        fArea_hull = np.sum(hull > 0)
        if (threshVal < 0.3) or (fArea_hull == 0):
            flag_fail_empty = 1
            break

        fArea, numObj, den, eccen, radius, intMean, intMean_perpx, hull_centroid = measure_genFeat(image,
                                                                                                   hull,
                                                                                                   fArea_hull,
                                                                                                   hull=True,
                                                                                                   flag_print=True)

        flag_fail_border = ROI_borderCheck(thresh)

        threshVals[idx][numTP - 1] = threshVal
        numObjs[idx][numTP - 1] = numObj
        fAreas[idx][numTP - 1] = fArea_hull
        dens[idx][numTP - 1] = den
        eccens[idx][numTP - 1] = eccen
        radii[idx][numTP - 1] = radius
        intMeans[idx][numTP - 1] = intMean

    elif (ch == 'Green') and (flag_fail_empty == 0):
        print(f'Now processing channel: Green')

        for tp in range(numTP):
            # Import 16 bit image.
            slice_idx = tp * 2 + 1 # 2 channels, SOX2 idx = 1
            sox2Dir = main_dir / 'Time-lapse' / f'array{arr}_raft{raft}_tl.tif'
            with tiff.TiffFile(sox2Dir) as tif:
                image = tif.pages[slice_idx].asarray()

            image_norm = rescale_intensity(image, in_range=(1500, 5000))

            # Create binary mask.
            thresh, threshVal = segment_germ(image, hull, flag_show=False)

            # Measure thresholded image.
            fArea, numObj, den, eccen, radius, intMean, intMean_perpx, _, regions = measure_genFeat(image,
                                                                                                    hull,
                                                                                                    fArea_hull,
                                                                                                    flag_print=True,
                                                                                                    flag_regions=True)

            numObjs[idx][tp] = numObj

            if numObj != 0:
                threshVals[idx][tp] = threshVal
                fAreas[idx][tp] = fArea
                dens[idx][tp] = den
                eccens[idx][tp] = eccen
                radii[idx][tp] = radius
                intMeans[idx][tp] = intMean
                intMeans_fold[tp] = intMean / intMeans[idx][0]
                intMeans_perpx[tp] = intMean_perpx
                intMeans_perpx_fold[tp] = intMean_perpx / intMeans_perpx[0]

                # Determine intMean from center normalized to intMean via concentric rings at tp = 0.
                if (flag_fail_border == 0) and (intMeans[idx][0] != None):
                    for count, r in enumerate(range(ringWidth, (gasRad + 1), ringWidth)):
                        for count, r in enumerate(range(ringWidth, (gasRad + 1), ringWidth)):
                            intMean_ring = fromCenter(image, hull, hull_centroid, r, ringWidth)
                            if math.isnan(intMean_ring):
                                break
                            intMean_fromCenter = intMean_ring / intMeans[idx][0]
                            ringNum = count
                            intMeans_fromCenter[tp][count] = intMean_fromCenter

                        intMean_radialDiff = intMean_fromCenter - intMeans_fromCenter[tp][0]
                        intMeans_radialSlope[tp] = intMean_radialDiff / (ringNum * ringWidth * pxSize)

                        # outDir = main_dir.parents[0] / 'From Center'
                        # outDir.mkdir(exist_ok=True)
                        # if tp == orderTP[-1]:
                        #     image *= masks[0]
                        #     plt.imshow(image, cmap='gray')
                        #     ring = ring[half_buff:half_buff+image.shape[0], half_buff:half_buff+image.shape[1]]
                        #     plt.imshow(ring, cmap='jet', alpha=0.5*(ring > 0))
                        #     plt.axis('off')
                        #     file_name = f'Array{arr}_Raft{raft}_TP{tp:02}_FromCenter_{r - 5}.jpg'
                        #     plt.savefig(str(outDir / file_name), bbox_inches='tight', pad_inches=0)

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

            fig.suptitle(f'array{arr}_raft{raft}_timepoint{tp:02}_SOX2', fontweight='bold')

            fig.tight_layout()

            plt.show()


print()

print(f'Flag_Empty: {flag_fail_empty}')
print(f'Flag_Border: {flag_fail_border}')

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
print(f'Intensity mean (radial slope): {intMeans_radialSlope}')