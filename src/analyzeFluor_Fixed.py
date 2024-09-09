# analyzeFluor_Fixed.py - Analyzes each gastruloid-on-raft.

from pathlib import Path
import os
from expMeta import ExpMeta
import itertools

import math
import pandas as pd
import numpy as np
from skimage import io
from skimage.morphology import closing, disk, remove_small_objects, convex_hull_image
from skimage.restoration import rolling_ball
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops_table


def analyzeFluor(exp, flag_finalTP=1):
    numRafts = exp.arrDim[2] ** 2
    channels = ['Blue', 'Green', 'Red', 'FarRed']

    # Import metadata csv.
    metaPath = Path(f'{exp.dataPath}_Analysis') / 'Experiment Metadata.csv'
    df_meta = pd.read_csv(str(metaPath))
    interval = df_meta['Values'].values[1]

    # Set num & order of time points depending on flag_finalTP.
    numTP = 1
    orderTP = list(range(numTP))
    if flag_finalTP == 0:
        numTP = df_meta['Values'].values[3]
        orderTP = [numTP - 1] + [*range(numTP - 1)]

    # Analyze each fluorescence image for each array.
    for arr in range(exp.numArrs):
        times = [None] * (numRafts * numTP)
        raftNums = [None] * (numRafts * numTP)

        flags_missing = [None] * (numRafts * numTP)
        flags_empty = [None] * (numRafts * numTP)
        flags_border = [None] * (numRafts * numTP)

        flags_lo_int = [0] * (numRafts * numTP)

        fAreas_hull = [None] * (numRafts * numTP)

        numObjs = [[None] * (numRafts * numTP) for i in range(len(channels))]
        threshVals = [[None] * (numRafts * numTP) for i in range(len(channels))]
        fAreas = [[None] * (numRafts * numTP) for i in range(len(channels))]
        dens = [[None] * (numRafts * numTP) for i in range(len(channels))]
        eccens = [[None] * (numRafts * numTP) for i in range(len(channels))]
        radii = [[None] * (numRafts * numTP) for i in range(len(channels))]

        count = 0  # keeps track of data row

        for raft in range(1, (numRafts + 1)):
            hull = None

            for tp in range(0, numTP):
                times[count] = orderTP[tp] * float(interval)
                raftNums[count] = raft
                thresh_vals = []

                for ch in range(len(channels)):
                    imagePath = Path(f'{exp.dataPath}_Analysis') / \
                                'Individual Raft Images' / \
                                f'array{arr + 1}_timepoint{orderTP[tp]:02}_{channels[ch]}'/ \
                                f'array{arr + 1}_timepoint{orderTP[tp]:02}_{channels[ch]}_raft{raft:03}.tif'

                    # Analyzes raft only if cropped raft image exists.
                    if os.path.isfile(imagePath):
                        flags_missing[count] = 0

                        print(f'\tExists: Raft {raft} @ Time Point {orderTP[tp]} in Channel {channels[ch]}')
                        image = io.imread(str(imagePath))

                        # # Crop out raft edges.
                        # cropCent = 0.085
                        # if channels[ch] == 'BF_ffc':
                        #     cropCent = 0.25
                        # cropDim = [image.shape[0] * cropCent,
                        #            image.shape[0] * (1 - cropCent),
                        #            image.shape[1] * cropCent,
                        #            image.shape[1] * (1 - cropCent)]
                        # cropDim = list(map(int, cropDim))
                        # image = image[cropDim[0]:cropDim[1], cropDim[2]:cropDim[3]]

                        if channels[ch] == 'Blue':
                            # Create binary mask.
                            thresh_val = 1500
                            thresh = image > thresh_val
                            hull = closing(thresh, disk(5))
                            hull = remove_small_objects(hull, min_size=25000)
                            hull = convex_hull_image(hull)

                            # Check for empty raft.
                            fArea_hull = np.sum(hull > 0)
                            fAreas_hull[count] = fArea_hull
                            if fArea_hull == 0:
                                flags_empty[count] = 1
                                print('\t\tEmpty raft!')
                                break
                            else:
                                flags_empty[count] = 0


                            # Measure hull image.
                            labels = label(hull)
                            props = regionprops_table(labels, properties=('centroid',
                                                                          'orientation',
                                                                          'axis_major_length',
                                                                          'axis_minor_length',
                                                                          'eccentricity',
                                                                          'bbox'))
                            df_measure = pd.DataFrame(props)

                            # Apply convex hull as mask to original thresholded Hoechst image.
                            thresh = np.where(hull, thresh, 0).astype(int)

                            # Determine numObj, fArea, den, eccen, and radius.
                            numObj = df_measure.shape[0]
                            fArea = np.sum(thresh > 0)
                            den = fArea / fAreas_hull[count]
                            if numObj == 1:
                                eccen = df_measure.loc[0]['eccentricity']
                                radius = ((df_measure.loc[0]['axis_major_length'] +
                                           df_measure.loc[0]['axis_minor_length']) / 2) / 2
                            else:
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
                                flags_border[count] = 1
                                print('\t\tTouching border!')
                            else:
                                flags_border[count] = 0

                        else:
                            if ch == 'Red' or ch == 'FarRed':
                                image = image - rolling_ball(image, radius=50)
                                image = gaussian(image, sigma=2, preserve_range=True)

                            # Create binary mask.
                            thresh_val = threshold_otsu(image)
                            thresh = image > thresh_val
                            thresh_vals.append(thresh_val)

                            # Apply complete hull to other channels.
                            thresh = np.where(hull, thresh, 0).astype(int)

                            # Measure thresholded image.
                            labels = label(thresh)
                            props = regionprops_table(labels, properties=('centroid',
                                                                          'orientation',
                                                                          'axis_major_length',
                                                                          'axis_minor_length',
                                                                          'eccentricity',
                                                                          'num_pixels'))
                            df_measure = pd.DataFrame(props)

                            numObj = df_measure.shape[0]
                            fArea = np.sum(thresh > 0)
                            den = fArea / fAreas_hull[count]
                            if numObj != 0:
                                df_measure = df_measure.sort_values(by='num_pixels', ascending=False, ignore_index=True)
                                eccen = df_measure.loc[0]['eccentricity']
                                radius = ((df_measure.loc[0]['axis_major_length'] +
                                           df_measure.loc[0]['axis_minor_length']) / 2) / 2
                            else:
                                eccen = None
                                radius = None

                            # # Only measure eccen and radius of single, large object.
                            # df_measure = df_measure.sort_values(by='num_pixels', ascending=False, ignore_index=True)
                            # # if (numObj == 1) or (numObj > 0 and df_measure.loc[0]['num_pixels'] / df_measure.loc[1][
                            #     'num_pixels'] > 5):
                            #     eccen = df_measure.loc[0]['eccentricity']
                            #     radius = ((df_measure.loc[0]['axis_major_length'] + df_measure.loc[0]['axis_minor_length']) / 2) / 2
                            # else:
                            #     eccen = None
                            #     radius = None

                        numObjs[ch][count] = numObj
                        threshVals[ch][count] = thresh_val
                        fAreas[ch][count] = fArea
                        dens[ch][count] = den
                        eccens[ch][count] = eccen
                        radii[ch][count] = radius

                    else:
                        print(f'\tDNE: Raft {raft} @ Time Point {orderTP[tp]}')
                        flags_missing[count] = 1
                        break

                if (flags_missing[count] == 0) and (flags_empty[count] == 0):
                    lo_int = [thresh_vals[0] < 200, thresh_vals[1] < 300, thresh_vals[2] < 200]
                    # print(f'\t{True in lo_int}')
                    if (True in lo_int):
                        flags_lo_int[count] = 1
                count += 1

        # Create data frame for Array 'arr'.
        df_rafts = pd.DataFrame({"Time": times,
                                 "Raft": raftNums,
                                 "Missing": flags_missing,
                                 "Empty": flags_empty,
                                 "Border": flags_border,
                                 "Lo_Int": flags_lo_int,
                                 "FAreaHull": fAreas_hull})
        for ch in range(len(channels)):
            df_rafts[f'NumObj_{channels[ch]}'] = numObjs[ch]
            df_rafts[f'ThreshVal_{channels[ch]}'] = threshVals[ch]
            df_rafts[f'FArea_{channels[ch]}'] = fAreas[ch]
            df_rafts[f'Dens_{channels[ch]}'] = dens[ch]
            df_rafts[f'Eccen_{channels[ch]}'] = eccens[ch]
            df_rafts[f'Radius_{channels[ch]}'] = radii[ch]

        df_rafts = df_rafts.sort_values(by=['Time', 'Raft'], ignore_index=True)

        # Save data frame for specific Array 'arr'.
        outPath_fanal = Path(f'{exp.dataPath}_Analysis') / f'array{arr + 1}_FluorAnalysis_Fixed.csv'
        df_rafts.to_csv(outPath_fanal, header=True, index=False)

        # Grade good/bad rafts.
        raft_grad(exp, arr, df_rafts)

        print(f'Saved fluor data frame for Array {arr + 1}.')

    print('Fluorescence analysis complete.')


# Grades rafts based on flags and saves separate dataframe.
def raft_grad(exp, arr, df_rafts):
    numRafts = exp.arrDim[2] ** 2
    excl_missing = []
    excl_empty = []
    excl_border = []
    excl_lo_int = []
    # fail_largeCen = []
    fail_eccen = []

    finalTP = df_rafts[df_rafts['Time'] == ((exp.numTimePoints - 1) * exp.interval)].reset_index(drop=True)
    for raft in range(1, (numRafts + 1)):
        # radii = [finalTP['Radius_Green'][raft - 1], finalTP['Radius_Red'][raft - 1], finalTP['Radius_FarRed'][raft - 1]]
        # eccens = [finalTP['Eccen_Green'][raft - 1], finalTP['Eccen_Red'][raft - 1], finalTP['Eccen_FarRed'][raft - 1]]

        # Raft has to be present
        if finalTP['Missing'][raft - 1] == 1:
            excl_missing.append(int(raft))
        elif finalTP['Empty'][raft - 1] == 1:
            excl_empty.append(int(raft))
        elif finalTP['Border'][raft - 1] == 1:
            excl_border.append(int(raft))
        elif finalTP['Lo_Int'][raft - 1] == 1:
            excl_lo_int.append(int(raft))
        # elif any(math.isnan(rad) for rad in radii):
        #     fail_largeCen.append(int(raft))
        elif finalTP['Eccen_Blue'][raft - 1] > 0.55:
            fail_eccen.append(raft)

    raftExcl = excl_missing + excl_empty + excl_border + excl_lo_int
    raftFail = fail_eccen
    # raftFail = fail_largeCen + fail_eccen
    raftBad = raftExcl + raftFail
    raftPass = [raft for raft in range(1, (numRafts + 1)) if raft not in raftBad]

    # Pad each list to proper length.
    grades = [raftPass, raftFail, raftExcl,
              fail_eccen,
              excl_missing, excl_empty, excl_border, excl_lo_int]
    # grades = [raftPass, raftFail, raftExcl,
    #           fail_largeCen, fail_eccen,
    #           excl_missing, excl_empty, excl_border]
    for grade in grades:
        grade.sort()
    # print(grades)
    grades = list(zip(*itertools.zip_longest(*grades, fillvalue=None)))

    df_grad = pd.DataFrame({"Pass": grades[0],
                            "Fail": grades[1],
                            "Exclude": grades[2],
                            "Fail_eccen": grades[3],
                            "Excl_missing": grades[4],
                            "Excl_empty": grades[5],
                            "Excl_border": grades[6],
                            "Excl_lo_int": grades[7]
                            })
    # df_grad = pd.DataFrame({"Pass": grades[0],
    #                         "Fail": grades[1],
    #                         "Exclude": grades[2],
    #                         "Fail_largeCen": grades[3],
    #                         "Fail_eccen": grades[4],
    #                         "Excl_missing": grades[5],
    #                         "Excl_empty": grades[6],
    #                         "Excl_border": grades[7]})

    outPath_grad = Path(f'{exp.dataPath}_Analysis') / f'array{arr + 1}_RaftGrading_Fixed.csv'
    df_grad.to_csv(outPath_grad, header=True, index=False)


# Run analysis without running raftImageAnalysis_main.py.
test = ExpMeta(dir=r'C:\Users\MVX\Documents\2 - Data\231019_RUES2_1.8e6_48hrBMP4_0%aneu_3',
               timeInt=99)  # replace parameters to fit experiment
test.numArrs = 1
test.numTimePoints = 1  # including Time Point 0
analyzeFluor(test)