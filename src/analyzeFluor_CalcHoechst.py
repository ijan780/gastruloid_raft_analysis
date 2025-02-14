
# analyzeFluor_CalcHoechst.py - Analyzes each gastruloid-on-raft.

from pathlib import Path
import os
from expMeta_EGOMRA import ExpMeta
import itertools

import pandas as pd
import numpy as np
from numpy import quantile
from skimage import io
from skimage.morphology import closing, remove_small_objects, convex_hull_image, disk
from skimage.filters import threshold_li
from skimage.measure import label, regionprops_table


def analyzeFluor(exp, flag_finalTP=1):
    numRafts = exp.arrDim[2] ** 2
    channels = ['Green', 'Blue']

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

        numObjs = [[None] * (numRafts * numTP) for i in range(len(channels))]

        fAreas_hull = [None] * (numRafts * numTP)
        eccens = [None] * (numRafts * numTP)
        radii = [None] * (numRafts * numTP)

        fAreas_nuc = [None] * (numRafts * numTP)
        NormDNAs = [None] * (numRafts * numTP)

        count = 0  # keeps track of data row

        print(f'Processing Array {arr + 1}')

        for raft in range(1, (numRafts + 1)):

            hull = None

            for tp in range(0, numTP):
                times[count] = orderTP[tp] * float(interval)
                raftNums[count] = raft

                for ch in range(len(channels)):
                    imagePath = Path(f'{exp.dataPath}_Analysis') / \
                                'Individual Raft Images' / \
                                f'array{arr + 1}_timepoint{orderTP[tp]:02}_{channels[ch]}_raft{raft:03}.tif'

                    # Analyzes raft only if cropped raft image exists.
                    if os.path.isfile(imagePath):
                        flags_missing[count] = 0
                        print(f'\tExists: Raft {raft} @ Time Point {orderTP[tp]} in Channel {channels[ch]}')
                        image = io.imread(str(imagePath))

                        # # Crop out raft edges.
                        # cropCent = 0.085
                        # cropDim = [image.shape[0] * cropCent,
                        #            image.shape[0] * (1 - cropCent),
                        #            image.shape[1] * cropCent,
                        #            image.shape[1] * (1 - cropCent)]
                        # cropDim = list(map(int, cropDim))
                        # image = image[cropDim[0]:cropDim[1], cropDim[2]:cropDim[3]]

                        if channels[ch] == 'Green':
                            # Create binary mask & check for empty raft.
                            thresh_value = threshold_li(image)
                            if thresh_value < 2000:
                                flags_empty[count] = 1
                                print('\tEmpty raft!')
                                break
                            else:
                                flags_empty[count] = 0
                            thresh = image > thresh_value

                            # Create convex hull surrounding entire gastruloid.
                            hull = closing(thresh, disk(5))
                            hull = remove_small_objects(hull, min_size=25000)
                            hull = convex_hull_image(hull)

                            # Check for empty raft again.
                            fArea_hull = np.sum(hull > 0)
                            fAreas_hull[count] = fArea_hull
                            if fArea_hull == 0:
                                flags_empty[count] = 1
                                print('\tEmpty raft!')
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

                            # Apply convex hull as mask to original thresholded Calcein AM image.
                            thresh = np.where(hull, thresh, 0).astype(int)

                            # Determine numObj, NormDNA, eccen, and radius.
                            numObj = df_measure.shape[0]
                            if numObj == 1:
                                eccen = df_measure.loc[0]['eccentricity']
                                radius = ((df_measure.loc[0]['axis_major_length'] +
                                           df_measure.loc[0]['axis_minor_length']) / 2) / 2
                                eccens[count] = eccen
                                radii[count] = radius

                            # Check if bounding box of thresh touches edge of raft.
                            bbox = [0, 0, 0, 0]
                            rows = np.any(thresh, axis=0)
                            cols = np.any(thresh, axis=1)
                            bbox[0], bbox[2] = np.where(rows)[0][[0, -1]]
                            bbox[1], bbox[3] = np.where(cols)[0][[0, -1]]

                            if (bbox[0] < (thresh.shape[0] * 0.01) or bbox[1] < (thresh.shape[1] * 0.01)) or (bbox[2] > (thresh.shape[0] * 0.99) or bbox[3] > (thresh.shape[1] * 0.99)):
                                flags_border[count] = 1
                                print('\tTouching border!')
                            else:
                                flags_border[count] = 0

                        elif channels[ch] == 'Blue':
                            # Create binary mask.
                            thresh = image > 6000

                            #  Apply convex hull as mask to thresholded image.
                            thresh = np.where(hull, thresh, 0).astype(int)
                            # Measure thresholded image.
                            labels = label(thresh)
                            props = regionprops_table(labels, properties=('centroid',
                                                                          'orientation',
                                                                          'axis_major_length',
                                                                          'axis_minor_length',
                                                                          'eccentricity'))
                            df_measure = pd.DataFrame(props)

                            numObj = df_measure.shape[0]
                            fArea_DNA = np.sum(thresh > 0)
                            NormDNA = fArea_DNA / fAreas_hull[count]

                            fAreas_nuc[count] = fArea_DNA
                            NormDNAs[count] = NormDNA

                        numObjs[ch][count] = numObj

                    else:
                        print(f'\tDNE: Raft {raft} @ Time Point {orderTP[tp]}')
                        flags_missing[count] = 1
                        break

                count += 1

        # Create data frame for Array 'arr'.
        df_rafts_raw = pd.DataFrame({"Time": times,
                                 "Raft": raftNums,
                                 "Missing": flags_missing,
                                 "Empty": flags_empty,
                                 "Border": flags_border,
                                 "NumObj_Green": numObjs[0],
                                 "FArea_Hull": fAreas_hull,
                                 "Eccen": eccens,
                                 "Radius": radii,
                                 "NumbObj_Blue": numObjs[1],
                                 "fArea_DNA": fAreas_nuc,
                                 "NormDNA": NormDNAs})

        df_rafts_raw = df_rafts_raw.sort_values(by=['Time', 'Raft'], ignore_index=True)

        # Save data frame for specific Array 'arr'.
        outFile_fAnal_raw = f'array{arr + 1}_FluorAnalysis_CalcHoechst_Raw.csv'
        outPath_fAnal_raw = Path(f'{exp.dataPath}_Analysis') / outFile_fAnal_raw
        df_rafts_raw.to_csv(outPath_fAnal_raw, header=True, index=False)

        df_rafts_filter = df_rafts_raw[(df_rafts_raw['Missing'] == 0)
                                       & (df_rafts_raw['Empty'] == 0)
                                       & (df_rafts_raw['Border'] == 0)]
        df_rafts_filter['NormDNA_Quartile'] = pd.cut(df_rafts_filter['NormDNA'], bins=4,
                                                    labels=['Q1', 'Q2', 'Q3', 'Q4'])
        df_rafts_filter = df_rafts_filter.sort_values(by=['NormDNA'])
        outFile_fAnal = f'array{arr + 1}_FluorAnalysis_CalcHoechst.csv'
        outPath_fAnal = Path(f'{exp.dataPath}_Analysis') / outFile_fAnal
        df_rafts_filter.to_csv(outPath_fAnal, header=True, index=False)

        NormDNAs_sorted = df_rafts_filter['NormDNA']
        df_quart = pd.DataFrame({"Quartile": ['Min', '1', '2', '3', 'Max'],
                                 "Value": quantile(NormDNAs_sorted, [0, 0.25, 0.5, 0.75, 1])})
        outFile_quart = f'array{arr + 1}_FluorAnalysis_CalcHoechst_Quart.csv'
        outPath_quart = Path(f'{exp.dataPath}_Analysis') / outFile_quart
        df_quart.to_csv(outPath_quart, header=True, index=False)

        # Grade good/bad rafts.
        raft_grad(exp, arr, df_rafts_raw)

        print(f'Saved fluor data frame for Array {arr + 1}.')

    print('Fluorescence analysis complete.')


# Grades rafts based on flags and saves separate dataframe.
def raft_grad(exp, arr, df_rafts):
    numRafts = exp.arrDim[2] ** 2
    excl_missing = []
    excl_empty = []
    excl_border = []
    fail_eccen = []

    finalTP = df_rafts[df_rafts['Time'] == ((exp.numTimePoints - 1) * exp.interval)].reset_index(drop=True)

    for raft in range(1, (numRafts + 1)):
        if finalTP['Missing'][raft - 1] == 1:
            excl_missing.append(int(raft))
        elif finalTP['Empty'][raft - 1] == 1:
            excl_empty.append(int(raft))
        elif finalTP['Border'][raft - 1] == 1:
            excl_border.append(int(raft))
        elif finalTP['Eccen'][raft - 1] > 0.55:
            fail_eccen.append(raft)

    raftExcl = excl_missing + excl_empty + excl_border
    raftFail = fail_eccen
    raftBad = raftExcl + raftFail
    raftPass = [raft for raft in range(1, (numRafts + 1)) if raft not in raftBad]


    # Pad each list to proper length.
    grades = [raftPass, raftFail, raftExcl,
              fail_eccen,
              excl_missing, excl_empty, excl_border]
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
                            "Excl_border": grades[6]})

    outPath_grad = Path(f'{exp.dataPath}_Analysis') / f'array{arr + 1}_RaftGrading_CalcHoechst.csv'
    df_grad.to_csv(outPath_grad, header=True, index=False)

if __name__ == '__main__':
    # Run analysis without running raftImageAnalysis_main.py.
    test = ExpMeta(dir=r'',
                   timeInt=99)  # replace parameters to fit experiment
    test.numArrs = 2
    test.numTimePoints = 1  # including Time Point 0
    analyzeFluor(test)