# raftImageAnalysis_main.py - Initializes key parameters for MVX raft image analysis.

from pathlib import Path
import os
import pandas as pd
from parse import parse

from expMeta import ExpMeta
from stitchArray import stitchArray
from cropRafts import cropRafts
from analyzeFluor_CalcHoechst import analyzeFluor

# Load or create new experiment metadata object.
load_or_create = input('Load new experiment? (y/n): ')
if load_or_create == 'y':
    metaDir = input('Enter data folder absolute path: ')
    metaDir = Path(f'{metaDir}_Analysis') / 'Experiment Metadata.csv'
    df_meta = pd.read_csv(metaDir)
    dict_meta = df_meta.set_index('Properties')['Values'].to_dict()

    dataDir = dict_meta['Data Dir']
    channels = dict_meta['Channels (total)']
    channel_mask = dict_meta['Channel (mask)']
    channels_tl = dict_meta['Channels (tl)'].strip('][').replace("'", "").split(', ')
    interval = float(dict_meta['Time Interval (hrs)'])
    maskDirSame = (dict_meta['Data Dir'] == dict_meta['Mask Dir'])
    numArrs = int(dict_meta['Number of Arrays'])
    numTimePoints = int(dict_meta['Number of Time Points'])

    # Create experiment metadata object.
    currExp = ExpMeta(dataDir=dataDir,
                      channels=channels,
                      channel_mask=channel_mask,
                      channels_tl=channels_tl,
                      timeInt=interval,
                      maskDirSame=maskDirSame,
                      numArrs=numArrs,
                      numTimePoints=numTimePoints)

elif load_or_create == 'n':
    # Request user input for data folder absolute path, mask path (same or different), mask channel, time intervals, and time-lapse channels.
    dataDir = input('Enter data folder absolute path: ')
    maskDirSame = input('Is the mask folder the same as the data folder? (y/n): ')
    if maskDirSame == 'y':
        maskDirSame = True
    elif maskDirSame == 'n':
        maskDirSame = False
    channel_mask = input('Enter channel for mask (b, v, g, r, fr): ')
    interval = float(input('Enter duration of time interval (hr): '))
    channels_tl = input('Enter time-series channels separated by spaces (all, none, bf, b, v, g, r, fr): ')
    channels_tl = channels_tl.split()

    # Determine number of arrays.
    numArrs = len(os.listdir(dataDir)) - 1 # account for "data" subfolder
    print(f'Number of arrays: {numArrs}')

    # Folder for first subsection.
    randomImgFolder = Path(dataDir) / '1' / '1x1'

    # Determine number of time points.
    pattern = '{timePoint}_{channel}.tif'
    lastImg = os.listdir(randomImgFolder)[-1]   # last image in folder for first subsection
    patternResult = parse(pattern, lastImg)
    numTimePoints = int(patternResult['timePoint']) + 1
    print(f'Number of time points: {numTimePoints}')

    # List all channels.
    channels = ['BF']
    for filename in os.listdir(randomImgFolder):
        patternResult = parse(pattern, filename)
        if patternResult['channel'] not in channels:
            channels.append(patternResult['channel'])
    print(f'Channels ({len(channels)} total): {channels}')

    # Create experiment metadata object.
    currExp = ExpMeta(dataDir=dataDir,
                      channels=channels,
                      channel_mask=channel_mask,
                      channels_tl=channels_tl,
                      timeInt=interval,
                      maskDirSame=maskDirSame,
                      numArrs=numArrs,
                      numTimePoints=numTimePoints)

    print(f'Channel mask: {currExp.channel_mask}')
    print(f'Channels time-series ({len(currExp.channels_tl)} total): {currExp.channels_tl}')
    print(f'Channels endpoint ({len(currExp.channels_end)} total): {currExp.channels_end}')
    print('')

    # Print out metadata.
    currExp.write_metadata()

# Stitch images into composite.
print('NOW STITCHING...')
stitchArray(currExp)
print('')

# # Crops individual identified rafts.
# print('NOW CROPPING TO RAFTS...')
# cropRafts(currExp)
# print('')



# # Performs fluorescence analysis.
# print('NOW PERFORMING FLUORESCENCE ANALYSIS')
# analyzeFluor(currExp, flag_finalTP=0)
# print('')

print('PROCESSING COMPLETE.')