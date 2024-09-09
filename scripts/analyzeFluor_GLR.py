# analyzeFluor_GLR.py - Analyzes each gastruloid-on-raft.

from pathlib import Path
import os
from expMeta import ExpMeta
import itertools

import math
import pandas as pd
import numpy as np
from skimage import io
from skimage.morphology import closing, opening, disk, remove_small_objects, convex_hull_image
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops_table
from skimage.draw import ellipse


def analyzeFluor(exp):
    numRafts = exp.arrDim[2] ** 2
    channels = ['Hull', 'Green']
    # channels = ['Blue', 'Green']

    # Import metadata csv.
    metaPath = Path(f'{exp.dataPath}_Analysis') / 'Experiment Metadata.csv'
    df_meta = pd.read_csv(str(metaPath))
    interval = float(df_meta['Values'].values[1])
    numTP = int(df_meta['Values'].values[3])
    timeOffset = float(df_meta['Values'].values[4])
    pxSize = float(df_meta['Values'].values[-2])

    # For from center measurements.
    ringWidth = 10 # px
    gasRad = math.ceil((500 / 2 / pxSize) / ringWidth) * ringWidth  # px rounded to nearest multiple of ringWidth
    numRings = int(gasRad / ringWidth)

    # Analyze each fluorescence image for each array.
    for arr in range(exp.numArrs):
        times = []
        raftNums = []

        flags_missing = []
        flags_empty = []
        flags_border = []

        # flags_lo_int = [0] * (numRafts * numTP)

        numObjs = [[], []]
        threshVals = [[], []]
        fAreas = [[], []]
        dens = [[], []]
        eccens = [[], []]
        radii = [[], []]
        intMeans = [[], []]
        intMeans_fold = []
        intMeans_perpx = []
        intMeans_perpx_fold = []
        intMeans_fromCenter = [[] for i in range(numRings)]

        for raft in range(1, (numRafts + 1)):
            for tp in range(numTP):
                raftNums.append(raft)
                times.append((tp * interval) + timeOffset)

            # Dedicated lists for each raft to later append to array-wide lists above.
            numObjs_raft = [[None] * numTP, [None] * numTP]
            threshVals_raft = [[None] * numTP, [None] * numTP]
            fAreas_raft = [[None] * numTP, [None] * numTP]
            dens_raft = [[None] * numTP, [None] * numTP]
            eccens_raft = [[None] * numTP, [None] * numTP]
            radii_raft = [[None] * numTP, [None] * numTP]
            intMeans_raft = [[None] * numTP, [None] * numTP]
            intMeans_fold_raft = [None] * numTP
            intMeans_perpx_raft = [None] * numTP
            intMeans_perpx_fold_raft = [None] * numTP
            intMeans_fromCenter_raft = [[None] * numTP for i in range(numRings)]

            # Determine ROI hull.
            imagePath_hull = Path(f'{exp.dataPath}_Calcein_Analysis') / \
                             'Individual Raft Images' / \
                             f'array{arr + 1}_timepoint00_Violet' / \
                             f'array{arr + 1}_timepoint00_Violet_raft{raft:03}.tif'

            # imagePath_hull = Path(f'{exp.dataPath}_Analysis') / \
            #                  'Individual Raft Images' / \
            #                  f'array{arr + 1}_timepoint{(numTP - 1):02}_Blue' / \
            #                  f'array{arr + 1}_timepoint{(numTP - 1):02}_Blue_raft{raft:03}.tif'

            # Only process rafts that exist.
            if os.path.isfile(imagePath_hull):
                print(f'\tExists: Raft {raft}')
                for tp in range(numTP):
                    flags_missing.append(0)

                # Create binary mask.
                image = io.imread(str(imagePath_hull))
                gaus = gaussian(image, sigma=2, preserve_range=True)
                threshVal = threshold_otsu(gaus)
                thresh = gaus > threshVal
                hull = opening(thresh, disk(5))
                hull = remove_small_objects(hull, min_size=25000)
                hull = convex_hull_image(hull)

                # Check for empty raft.
                fArea_hull = np.sum(hull > 0)
                for tp in range(numTP):
                    fAreas_raft[0][tp] = fArea_hull

                if (threshVal < 1e4) or (fArea_hull == 0):
                    print('\t\tEmpty raft!')
                    for tp in range(numTP):
                        flags_empty.append(1)
                        flags_border.append(None)
                else:
                    for tp in range(numTP):
                        flags_empty.append(0)

                    # Apply convex hull as mask to original thresholded image.
                    thresh = np.where(hull, thresh, 0).astype(int)

                    # Measure hull image.
                    labels = label(hull)
                    props = regionprops_table(labels, properties=('centroid',
                                                                  'orientation',
                                                                  'axis_major_length',
                                                                  'axis_minor_length',
                                                                  'eccentricity',
                                                                  'bbox',
                                                                  'num_pixels'))
                    df_measure = pd.DataFrame(props)

                    # Determine numObj, den, eccen, radius, and intMean.
                    numObj = df_measure.shape[0]
                    den = 1
                    df_measure = df_measure.sort_values(by='num_pixels', ascending=False, ignore_index=True)
                    eccen = df_measure.loc[0]['eccentricity']
                    radius = ((df_measure.loc[0]['axis_major_length'] +
                              df_measure.loc[0]['axis_minor_length']) / 2) / 2
                    intMean = np.mean(image[thresh > 0])
                    hull_centroid = (df_measure.loc[0]['centroid-0'], df_measure.loc[0]['centroid-1'])

                    # Check if bounding box of thresh touches edge of raft.
                    bbox = [0, 0, 0, 0]
                    rows = np.any(thresh, axis=0)
                    cols = np.any(thresh, axis=1)
                    bbox[0], bbox[2] = np.where(rows)[0][[0, -1]]
                    bbox[1], bbox[3] = np.where(cols)[0][[0, -1]]

                    if (bbox[0] < (thresh.shape[0] * 0.01) or bbox[1] < (thresh.shape[1] * 0.01)) or (
                            bbox[2] > (thresh.shape[0] * 0.99) or bbox[3] > (thresh.shape[1] * 0.99)):
                        print('\t\tTouching border!')
                        flag_border = 1
                    else:
                        flag_border = 0

                    # Save features for ROI hull.
                    for tp in range(numTP):
                        flags_border.append(flag_border)
                        threshVals_raft[0][tp] = threshVal
                        numObjs_raft[0][tp] = numObj
                        dens_raft[0][tp] = den
                        eccens_raft[0][tp] = eccen
                        radii_raft[0][tp] = radius
                        intMeans_raft[0][tp] = intMean

                    # TODO: check if image exists at timepoint
                    for tp in range(numTP):
                        # Import SOX2 image.
                        imagePath = Path(f'{exp.dataPath}_Analysis') / \
                                    'Individual Raft Images' / \
                                    f'array{arr + 1}_timepoint{tp:02}_Green' / \
                                    f'array{arr + 1}_timepoint{tp:02}_Green_raft{raft:03}.tif'
                        image = io.imread(str(imagePath))

                        # Create binary mask.
                        gaus = gaussian(image, sigma=4, preserve_range=True)
                        threshVal = threshold_otsu(gaus)
                        thresh = gaus > threshVal

                        # Apply convex hull as mask to original thresholded image.
                        thresh = np.where(hull, thresh, 0).astype(int)
                        fArea = np.sum(thresh > 0)

                        # Measure thresholded image.
                        labels = label(thresh)
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
                        numObjs_raft[1][tp] = numObj
                        if numObj != 0:
                            df_measure = df_measure.sort_values(by='num_pixels', ascending=False, ignore_index=True)
                            nucDen = fArea / fArea_hull
                            eccen = df_measure.loc[0]['eccentricity']
                            radius = ((df_measure.loc[0]['axis_major_length'] +
                                      df_measure.loc[0]['axis_minor_length']) / 2) / 2
                            intMean = np.mean(image[thresh > 0])
                            intMean_perpx = intMean / fArea
                            threshVals_raft[1][tp] = threshVal

                            fAreas_raft[1][tp] = fArea
                            dens_raft[1][tp] = nucDen
                            eccens_raft[1][tp] = eccen
                            radii_raft[1][tp] = radius
                            intMeans_raft[1][tp] = intMean
                            intMeans_fold_raft[tp] = intMean / intMeans_raft[1][0]
                            intMeans_perpx_raft[tp] = intMean_perpx
                            intMeans_perpx_fold_raft[tp] = intMean_perpx / intMeans_perpx_raft[0]

                            # Determine intMean from center normalized to intMean via concentric rings at tp = 0.
                            if (flag_border == 0) and (intMeans[ch][0] != None):
                                count = 0
                                for r in range(ringWidth, (gasRad + 1), ringWidth):
                                    buff = 60  # px
                                    half_buff = int(buff / 2)  # px

                                    image_expand = np.zeros((image.shape[0] + buff, image.shape[1] + buff))
                                    image_expand[half_buff:half_buff + image.shape[0],
                                    half_buff:half_buff + image.shape[1]] = image
                                    hull_expand = np.zeros((image.shape[0] + buff, image.shape[1] + buff))
                                    hull_expand[half_buff:half_buff + image.shape[0],
                                    half_buff:half_buff + image.shape[1]] = hull

                                    ring = np.zeros((image.shape[0] + buff, image.shape[1] + buff))
                                    rr, cc = ellipse(hull_centroid[0] + half_buff, hull_centroid[1] + half_buff,
                                                     r, r)
                                    ring[rr, cc] = 1
                                    rr_s, cc_s = ellipse(hull_centroid[0] + half_buff, hull_centroid[1] + half_buff,
                                                         (r - ringWidth),
                                                         (r - ringWidth))
                                    ring[rr_s, cc_s] = 0

                                    # Only measure pixels in ROI hull.
                                    intMean_ring = np.mean(image_expand[(ring > 0) & (hull_expand > 0)])
                                    if math.isnan(intMean_ring):
                                        break
                                    intMeans_fromCenter_raft[count][tp] = intMean_ring / intMeans_raft[1][0]

                                    count += 1
            else:
                print(f'\tDNE: Raft {raft} @ Time Point {tp:02}')
                for tp in range(numTP):
                    flags_missing.append(1)
                    flags_empty.append(None)
                    flags_border.append(None)

            # Append raft-specific lists to array-wide lists.
            for ch in range(len(channels)):
                numObjs[ch] += numObjs_raft[ch]
                threshVals[ch] += threshVals_raft[ch]
                fAreas[ch] += fAreas_raft[ch]
                dens[ch] += dens_raft[ch]
                eccens[ch] += eccens_raft[ch]
                radii[ch] += radii_raft[ch]
                intMeans[ch] += intMeans_raft[ch]
            intMeans_fold += intMeans_fold_raft
            intMeans_perpx += intMeans_perpx_raft
            intMeans_perpx_fold += intMeans_perpx_fold_raft
            for r in range(numRings):
                intMeans_fromCenter[r] += intMeans_fromCenter_raft[r]

            # print(len(raftNums),
            #       len(times),
            #       len(numObjs[0]),
            #       len(threshVals[0]),
            #       len(fAreas[0]),
            #       len(dens[0]),
            #       len(eccens[0]),
            #       len(radii[0]),
            #       len(intMeans[0]),
            #       len(intMeans_perpx),
            #       len(intMeans_perpx_fold),
            #       len(intMeans_fold),
            #       len(intMeans_fromCenter[0])
            #       )
            # print(len(times),
            #       len(raftNums),
            #       len(flags_missing),
            #       len(flags_empty),
            #       len(flags_border))

        # Create data frame for Array 'arr'.
        df_rafts = pd.DataFrame({"Time": times,
                                 "Raft": raftNums,
                                 "Missing": flags_missing,
                                 "Empty": flags_empty,
                                 "Border": flags_border})
                                 # "Lo_Int": flags_lo_int)
        for ch in range(len(channels)):
            df_rafts[f'NumObj_{channels[ch]}'] = numObjs[ch]
            df_rafts[f'ThreshVal_{channels[ch]}'] = threshVals[ch]
            df_rafts[f'FArea_{channels[ch]}'] = fAreas[ch]
            df_rafts[f'Den_{channels[ch]}'] = dens[ch]
            df_rafts[f'Eccen_{channels[ch]}'] = eccens[ch]
            df_rafts[f'Radius_{channels[ch]}'] = radii[ch]
            df_rafts[f'IntMean_{channels[ch]}'] = intMeans[ch]
        df_rafts['IntMean_Fold_Green'] = intMeans_fold
        df_rafts['IntMean_PerPx_Green'] = intMeans_perpx
        df_rafts['IntMean_PerPx_Fold_Green'] = intMeans_perpx_fold
        for r in range(numRings):
            disFromCen = (ringWidth / 2) + (r * (ringWidth)) # px
            df_rafts[f'IntMean_fromCenter_{disFromCen}_Green'] = intMeans_fromCenter[r]

        df_rafts = df_rafts.sort_values(by=['Raft', 'Time'], ignore_index=True)

        # Save data frame for specific Array 'arr'.
        outPath_fanal = Path(f'{exp.dataPath}_Analysis') / f'array{arr + 1}_FluorAnalysis_GLR.csv'
        df_rafts.to_csv(outPath_fanal, header=True, index=False)

        # Grade good/bad rafts.
        raft_grad(exp, arr, df_rafts)

        print(f'Saved fluor data frame for Array {arr + 1}.')

    print('Fluorescence analysis complete.')


# Grades rafts based on flags and saves separate dataframe.
def raft_grad(exp, arr, df_rafts):
    numRafts = exp.arrDim[2] ** 2

    missing = df_rafts[df_rafts['Missing'] == 1]
    excl_missing = list(set(missing['Raft'].tolist()))
    empty = df_rafts[df_rafts['Empty'] == 1]
    excl_empty = list(set(empty['Raft'].tolist()))
    border = df_rafts[df_rafts['Border'] == 1]
    excl_border = list(set(border['Raft'].tolist()))

    raftExcl = excl_missing + excl_empty + excl_border
    raftPass = [raft for raft in range(1, (numRafts + 1)) if raft not in raftExcl]

    # Pad each list to proper length.
    grades = [raftPass, raftExcl,
              excl_missing, excl_empty, excl_border]
    for grade in grades:
        grade.sort()
    # print(grades)
    grades = list(zip(*itertools.zip_longest(*grades, fillvalue=None)))

    df_grad = pd.DataFrame({"Pass": grades[0],
                            "Exclude": grades[1],
                            "Excl_missing": grades[2],
                            "Excl_empty": grades[3],
                            "Excl_border": grades[4],
                            })

    outPath_grad = Path(f'{exp.dataPath}_Analysis') / f'array{arr + 1}_RaftGrading_GLR.csv'
    df_grad.to_csv(outPath_grad, header=True, index=False)


# Run analysis without running raftImageAnalysis_main.py.
test = ExpMeta(dir=r'C:\Users\MVX\Documents\2 - Data\240729_R2GLR_1.8e6_0-48hrBMP4_0%aneu_3')
analyzeFluor(test)