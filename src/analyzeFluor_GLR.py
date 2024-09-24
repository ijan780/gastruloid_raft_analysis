# analyzeFluor_GLR.py - Analyzes each gastruloid-on-raft (requires time-lapse imaging.

from src.readMeta import ReadMeta

import math
import tifffile as tiff
import pandas as pd
import numpy as np
from skimage import io
from skimage.exposure import equalize_adapthist
from skimage.morphology import closing, opening, disk, remove_small_holes, remove_small_objects, convex_hull_image
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops_table, regionprops
from skimage.draw import ellipse
import matplotlib.pyplot as plt
from itertools import zip_longest


def analyzeFluor(exp):
    numRafts = exp.arrDim[2] ** 2

    interval = exp.interval
    numTP = exp.numTimePoints
    timeOffset = exp.timeOffset
    pxSize = exp.pxSize

    channel_mask = exp.channel_mask
    mask_idx = exp.channels_end.index(channel_mask)
    ch_end = exp.channels_end
    if 'BF' in ch_end:
        ch_end.remove('BF')
    ch_tl = exp.channels_tl
    ch_tl.remove('BF')
    fluor_channels = ch_end + ch_tl

    # For from center measurements.
    ringWidth = 10  # px
    gasRad = math.ceil((500 / 2 / pxSize) / ringWidth) * ringWidth  # px rounded to nearest multiple of ringWidth
    numRings = int(gasRad / ringWidth)

    # Analyze each fluorescence image for each array.
    for arr in range(exp.numArrs):
        print(f'Processing Array {arr + 1}')
        raftNums = [raft for raft in range(1, (numRafts + 1)) for _ in range(numTP)]
        times = [((tp * interval) + timeOffset) for _ in range(1, (numRafts + 1)) for tp in range(numTP)]

        flags_missing = []
        flags_empty = []
        flags_border = []

        # flags_lo_int = [0] * (numRafts * numTP)

        # Features for all channels.
        numObjs = [[] for _ in fluor_channels]
        threshVals = [[] for _ in fluor_channels]
        fAreas = [[] for _ in fluor_channels]
        dens = [[] for _ in fluor_channels]
        eccens = [[] for _ in fluor_channels]
        radii = [[] for _ in fluor_channels]
        intMeans = [[] for _ in fluor_channels]

        # Features for only time-lapse channels.
        intMeans_fold = [[] for _ in ch_tl]
        intMeans_perpx = [[] for _ in ch_tl]
        intMeans_perpx_fold = [[] for _ in ch_tl]
        intMeans_fromCenter = [[[] for _ in range(numRings)] for _ in ch_tl]
        intMeans_radialSlope = [[] for _ in ch_tl]

        for raft in range(1, (numRafts + 1)):
            # Dedicated lists for each raft to later append to array-wide lists above.
            numObjs_raft = [[None] * numTP for _ in fluor_channels]
            threshVals_raft = [[None] * numTP for _ in fluor_channels]
            fAreas_raft = [[None] * numTP for _ in fluor_channels]
            dens_raft = [[None] * numTP for _ in fluor_channels]
            eccens_raft = [[None] * numTP for _ in fluor_channels]
            radii_raft = [[None] * numTP for _ in fluor_channels]
            intMeans_raft = [[None] * numTP for _ in fluor_channels]
            intMeans_fold_raft = [[None] * numTP for _ in ch_tl]
            intMeans_perpx_raft = [[None] * numTP for _ in ch_tl]
            intMeans_perpx_fold_raft = [[None] * numTP for _ in ch_tl]
            intMeans_fromCenter_raft = [[[None] * numTP for _ in range(numRings)] for _ in ch_tl]
            intMeans_radialSlope_raft = [[None] * numTP for _ in ch_tl]

            # Directory of endpoint image.
            dir_end = exp.analDir / 'Individual Raft Images' / 'Endpoint'
            imageDir_end = dir_end / f'array{arr + 1}_raft{raft:03}_end.tif'

            # Only process rafts that exist.
            if imageDir_end.is_file():
                print(f'\tExists: Raft {raft}')
                flags_missing += [0] * numTP

                # Create binary mask for ROI mask.
                with tiff.TiffFile(imageDir_end) as tif:
                    image = tif.pages[mask_idx].asarray()
                hull, thresh, threshVal = segment_hull(image)

                # Check for empty raft.
                fArea_hull = np.sum(hull > 0)

                if (threshVal < 0.3) or (fArea_hull == 0):
                    print('\t\tEmpty raft!')
                    flags_empty += [1] * numTP
                    flags_border += [None] * numTP

                else:
                    flags_empty += [0] * numTP

                    # Measure hull image.
                    _, numObj, den, eccen, radius, intMean, _, hull_centroid = measure_genFeat(image, hull,
                                                                                                fArea_hull,
                                                                                                hull=True)

                    # Check if bounding box of ROI thresh touches edge of raft.
                    flag_border = ROI_borderCheck(thresh)
                    flags_border += [flag_border] * numTP

                    # Save features for ROI hull.
                    numObjs_raft[mask_idx][exp.numTimePoints - 1] = numObj
                    fAreas_raft[mask_idx][exp.numTimePoints - 1] = fArea_hull # fArea refers to hull (not thresh) area
                    threshVals_raft[mask_idx][exp.numTimePoints - 1] = threshVal
                    dens_raft[mask_idx][exp.numTimePoints - 1] = den
                    eccens_raft[mask_idx][exp.numTimePoints - 1] = eccen
                    radii_raft[mask_idx][exp.numTimePoints - 1] = radius
                    intMeans_raft[mask_idx][exp.numTimePoints - 1] = intMean

                    # TODO: create binary mask for & process other endpoint channels here.

                    # Process time-lapse germ layer reporters.
                    for tp in range(numTP):
                        for idx, ch in enumerate(ch_tl):
                            slice_idx = tp * (len(ch_tl) + 1) + (ch_tl.index(ch) + 1)  # accounting for removal of BF channel


                            dir_tl = exp.analDir / 'Individual Raft Images' / 'Time-lapse'
                            imageDir_tl = dir_tl / f'array{arr + 1}_raft{raft:03}_tl.tif'

                            with tiff.TiffFile(imageDir_tl) as tif:
                                image = tif.pages[slice_idx].asarray()

                            # Create binary mask.
                            thresh, threshVal = segment_germ(image, hull)

                            # Measure thresholded image.
                            fArea, numObj, den, eccen, radius, intMean, intMean_perpx, _ = measure_genFeat(image,
                                                                                                           hull,
                                                                                                           fArea_hull)
                            numObjs_raft[idx + len(ch_end)][tp] = numObj

                            if numObj != 0:
                                threshVals_raft[idx + len(ch_end)][tp] = threshVal
                                fAreas_raft[idx + len(ch_end)][tp] = fArea
                                dens_raft[idx + len(ch_end)][tp] = den
                                eccens_raft[idx + len(ch_end)][tp] = eccen
                                radii_raft[idx + len(ch_end)][tp] = radius
                                intMeans_raft[idx + len(ch_end)][tp] = intMean
                                intMeans_fold_raft[idx][tp] = (intMean /
                                                               intMeans_raft[idx + len(ch_end)][0])
                                intMeans_perpx_raft[idx][tp] = intMean_perpx
                                intMeans_perpx_fold_raft[idx][tp] = intMean_perpx / intMeans_perpx_raft[idx][0]

                                # Determine intMean from center normalized to intMean via concentric rings at tp = 0.
                                if (flag_border == 0) and (intMeans_raft[idx + len(ch_end)][0] != None):
                                    for count, r in enumerate(range(ringWidth, (gasRad + 1), ringWidth)):
                                        intMean_ring = fromCenter(image, hull, hull_centroid, r, ringWidth)
                                        if math.isnan(intMean_ring):
                                            break
                                        intMean_fromCenter = intMean_ring / intMeans_raft[idx + len(ch_end)][0]
                                        ringNum = count
                                        intMeans_fromCenter_raft[idx][count][tp] =  intMean_fromCenter

                                    intMean_radialDiff = intMean_fromCenter - intMeans_fromCenter_raft[idx][0][tp]
                                    intMeans_radialSlope_raft[idx][tp] = (intMean_radialDiff /
                                                                          (ringNum * ringWidth * pxSize))

            else:
                print(f'\tDNE: Raft {raft}')
                flags_missing += [1] * numTP
                flags_empty += [None] * numTP
                flags_border += [None] * numTP

            # Append raft-specific lists to array-wide lists.
            for idx, _ in enumerate(fluor_channels):
                numObjs[idx] += numObjs_raft[idx]
                threshVals[idx] += threshVals_raft[idx]
                fAreas[idx] += fAreas_raft[idx]
                dens[idx] += dens_raft[idx]
                eccens[idx] += eccens_raft[idx]
                radii[idx] += radii_raft[idx]
                intMeans[idx] += intMeans_raft[idx]
            for idx, _ in enumerate(ch_tl):
                intMeans_fold[idx] += intMeans_fold_raft[idx]
                intMeans_perpx[idx] += intMeans_perpx_raft[idx]
                intMeans_perpx_fold[idx] += intMeans_perpx_fold_raft[idx]
                for r in range(numRings):
                    intMeans_fromCenter[idx][r] += intMeans_fromCenter_raft[idx][r]
                intMeans_radialSlope[idx] += intMeans_radialSlope_raft[idx]

            # print(f'\t\ttimes: {len(times)}')
            # print(f'\t\traftNums: {len(raftNums)}')
            # print(f'\t\tflags_missing: {len(flags_missing)}')
            # print(f'\t\tflags_empty: {len(flags_empty)}')
            # print(f'\t\tflags_border: {len(flags_border)}')
            # print(f'\t\tnumObjs_raft: {len(numObjs[1])}')
            # print(f'\t\tthreshVals_raft: {len(threshVals[1])}')
            # print(f'\t\tfAreas_raft: {len(fAreas[1])}')
            # print(f'\t\tdens_raft: {len(dens[1])}')
            # print(f'\t\teccens_raft: {len(eccens[1])}')
            # print(f'\t\tradii_raft: {len(radii[1])}')
            # print(f'\t\tintMeans_raft: {len(intMeans[1])}')
            # print(f'\t\tintMeans_fold: {len(intMeans_fold[0])}')
            # print(f'\t\tintMeans_perpx: {len(intMeans_perpx[0])}')
            # print(f'\t\tintMeans_perpx_fold: {len(intMeans_perpx_fold[0])}')
            # print(f'\t\tintMeans_fromCenter: {len(intMeans_fromCenter[0][0])}')
            # print(f'\t\tintMeans_radialSlope: {len(intMeans_radialSlope[0])}')

        # Create data frame for Array 'arr'.
        df_rafts = saveFluorAnal(exp, ch_end, ch_tl, arr, raftNums, times,
                                 flags_missing, flags_empty, flags_border,
                                 numObjs, threshVals, fAreas, dens, eccens, radii, intMeans,
                                 intMeans_fold, intMeans_perpx, intMeans_perpx_fold,
                                 numRings, ringWidth, intMeans_fromCenter, intMeans_radialSlope)

        # Grade good/bad rafts.
        raft_grad(exp, arr, df_rafts)

        print(f'Saved fluor data frame for Array {arr + 1}.')
        print()

    print('Fluorescence analysis complete.')


def segment_hull(image, flag_show=False):
    image = equalize_adapthist(image)
    gaus = gaussian(image, sigma=2, preserve_range=True)
    threshVal = threshold_otsu(gaus)
    thresh = gaus > threshVal
    opened = opening(thresh, disk(5))
    filled = remove_small_holes(opened, area_threshold=5e4)
    largest_region = keep_largest(filled)
    # hull = remove_small_objects(hull, min_size=7.5e4)
    hull = convex_hull_image(largest_region)
    thresh = np.where(hull, thresh, 0).astype(int)

    if flag_show:
        io.imshow(thresh)
        plt.show()
        io.imshow(largest_region)
        plt.show()
        io.imshow(hull)
        plt.show()

    return hull, thresh, threshVal


def keep_largest(mask):
    labels = label(mask)
    regions = regionprops(labels)
    largest_region = max(regions, key=lambda r: r.area)
    largest_region_mask = np.zeros_like(labels, dtype=bool)
    largest_region_mask[labels == largest_region.label] = True

    return largest_region_mask


def segment_germ(image, hull, flag_show = False):
    image = equalize_adapthist(image)
    gaus = gaussian(image, sigma=4, preserve_range=True)
    threshVal = threshold_otsu(gaus)
    thresh = gaus > threshVal
    thresh_masked = np.where(hull, thresh, 0).astype(int)  # apply hull as mask to original thresholded image

    if flag_show:
        io.imshow(thresh)
        plt.show()
        io.imshow(thresh_masked)
        plt.show()

    return thresh_masked, threshVal


def measure_genFeat(image, thresh, fArea_hull, hull=False, flag_print=False, flag_regions=False):
    labels = label(thresh)
    props = regionprops_table(labels, properties=('centroid',
                                                  'orientation',
                                                  'axis_major_length',
                                                  'axis_minor_length',
                                                  'eccentricity',
                                                  'bbox',
                                                  'num_pixels'))
    df_measure = pd.DataFrame(props)
    if flag_print:
        print(df_measure)

    numObj = df_measure.shape[0]
    fArea = np.sum(thresh > 0)
    if numObj != 0:
        df_measure = df_measure.sort_values(by='num_pixels', ascending=False, ignore_index=True)
        eccen = df_measure.loc[0]['eccentricity']
        radius = ((df_measure.loc[0]['axis_major_length'] +
                   df_measure.loc[0]['axis_minor_length']) / 2) / 2
        intMean = np.mean(image[thresh > 0])
        intMean_perpx = intMean / fArea
    else:
        eccen = None
        radius = None
        intMean = None
        intMean_perpx = None

    if hull:
        hull_centroid = (df_measure.loc[0]['centroid-0'], df_measure.loc[0]['centroid-1'])
        den = 1
    else:
        hull_centroid = None
        den = fArea / fArea_hull

    if flag_regions:
        regions = regionprops(labels)
        return fArea, numObj, den, eccen, radius, intMean, intMean_perpx, hull_centroid, regions
    else:
        return fArea, numObj, den, eccen, radius, intMean, intMean_perpx, hull_centroid


def ROI_borderCheck(thresh):
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

    return flag_border


def fromCenter(image, hull, hull_centroid, r, ringWidth):
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

    return intMean_ring


# Creates and saves dataframe of raft data.
def saveFluorAnal(exp, ch_end, ch_tl, arr, times, raftNums,
                  flags_missing, flags_empty, flags_border,
                  numObjs, threshVals, fAreas, dens, eccens, radii, intMeans,
                  intMeans_fold, intMeans_perpx, intMeans_perpx_fold,
                  numRings, ringWidth, intMeans_fromCenter, intMeans_radialSlope):

    rep_and_stain = exp.reporters_and_stains

    df_rafts = pd.DataFrame({'Raft': raftNums,
                             'Time': times,
                             'Missing': flags_missing,
                             'Empty': flags_empty,
                             'Border': flags_border})

    for idx, ch in enumerate(ch_end):
        rep_or_stain = rep_and_stain[f'end_{ch}']

        df_rafts[f'NumObj_{rep_or_stain}'] = numObjs[idx]
        df_rafts[f'ThreshVal_{rep_or_stain}'] = threshVals[idx]
        df_rafts[f'FArea_{rep_or_stain}'] = fAreas[idx]
        df_rafts[f'Den_{rep_or_stain}'] = dens[idx]
        df_rafts[f'Eccen_{rep_or_stain}'] = eccens[idx]
        df_rafts[f'Radius_{rep_or_stain}'] = radii[idx]
        df_rafts[f'IntMean_{rep_or_stain}'] = intMeans[idx]

    for idx, ch in enumerate(ch_tl):
        rep_or_stain = rep_and_stain[f'tl_{ch}']

        df_rafts[f'NumObj_{rep_or_stain}'] = numObjs[idx + len(ch_end)]
        df_rafts[f'ThreshVal_{rep_or_stain}'] = threshVals[idx + len(ch_end)]
        df_rafts[f'FArea_{rep_or_stain}'] = fAreas[idx + len(ch_end)]
        df_rafts[f'Den_{rep_or_stain}'] = dens[idx + len(ch_end)]
        df_rafts[f'Eccen_{rep_or_stain}'] = eccens[idx + len(ch_end)]
        df_rafts[f'Radius_{rep_or_stain}'] = radii[idx+ len(ch_end)]
        df_rafts[f'IntMean_{rep_or_stain}'] = intMeans[idx + len(ch_end)]

        df_rafts[f'IntMean_Fold_{rep_or_stain}'] = intMeans_fold[idx]
        df_rafts[f'IntMean_PerPx_{rep_or_stain}'] = intMeans_perpx[idx]
        df_rafts[f'IntMean_PerPx_{rep_or_stain}'] = intMeans_perpx_fold[idx]
        for r in range(numRings):
            disFromCen = (ringWidth / 2) + (r * (ringWidth))  # px
            df_rafts[f'IntMean_fromCenter_{disFromCen}_{rep_or_stain}'] = intMeans_fromCenter[idx][r]
        df_rafts[f'IntMean_RadialSlope_{rep_or_stain}'] = intMeans_radialSlope[idx]

    df_rafts = df_rafts.sort_values(by=['Raft', 'Time'], ignore_index=True)

    outDir_fanal = exp.analDir / f'array{arr + 1}_FluorAnalysis_GLR.csv'
    df_rafts.to_csv(outDir_fanal, header=True, index=False)

    return df_rafts


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
    grades = list(zip(*zip_longest(*grades, fillvalue=None)))

    df_grad = pd.DataFrame({"Pass": grades[0],
                            "Exclude": grades[1],
                            "Excl_missing": grades[2],
                            "Excl_empty": grades[3],
                            "Excl_border": grades[4],
                            })

    outPath_grad = exp.analDir / f'array{arr + 1}_RaftGrading_GLR.csv'
    df_grad.to_csv(outPath_grad, header=True, index=False)


# Run analysis without running raftImageAnalysis_main.py.
if __name__ == '__main__':
    dataDir = r'C:\Users\MVX\Documents\2 - Data\240805_R2GLR_1.8e6_0-48hrBMP4_0%aneu_4'
    currExp = ReadMeta(dataDir)

    analyzeFluor(currExp)
