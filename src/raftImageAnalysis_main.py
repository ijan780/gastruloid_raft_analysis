# raftImageAnalysis_main.py - Initializes key parameters for MVX raft image analysis.

from expMeta import ExpMeta
from readMeta import ReadMeta
from stitchArray import stitchArray
from cropRafts import cropRafts
from analyzeFluor_CalcHoechst import analyzeFluor

# Load or create new experiment metadata object.
while True:
    load_or_create = input('Load experiment? (y/n): ')
    print()
    if load_or_create.lower() == 'y':
        print('LOADING EXPERIMENT...')
        dataDir = input('Enter data folder absolute path: ')
        currExp = ReadMeta(dataDir)
        break

    elif load_or_create.lower() == 'n':
        print('CREATING EXPERIMENT...')
        # Request user input for data folder absolute path, mask path (same or different), mask channel, time intervals, and reporter channels.
        dataDir = input('Enter data folder absolute path: ')
        while True:
            endDirSame = input('Is the mask folder the same as the data folder? (y/n): ')
            if endDirSame.lower() == 'y':
                endDirSame = True
                break
            elif endDirSame.lower() == 'n':
                endDirSame = False
                break
            else:
                continue
        channel_mask = input('Enter channel for mask (b, v, g, r, fr): ')
        interval = float(input('Enter duration of time interval (hr): '))
        timeOffset = float(input('Enter time offset (hr): '))
        channels_tl = input('Enter reporter channels separated by spaces (all, none, b, v, g, r, fr): ')
        channels_tl = channels_tl.split()
        print()

        # Create experiment metadata object.
        currExp = ExpMeta(dataDir=dataDir,
                          channel_mask=channel_mask,
                          channels_tl=channels_tl,
                          timeInt=interval,
                          endDirSame=endDirSame,
                          timeOffset=timeOffset)

        print(f'Channel mask: {currExp.channel_mask}')
        print(f'Channels time-series ({len(currExp.channels_tl)} total): {currExp.channels_tl}')
        print(f'Channels endpoint ({len(currExp.channels_end)} total): {currExp.channels_end}')
        print()
        currExp.set_endStains()
        print()

        # Print out metadata.
        currExp.write_metadata()

        break
    else:
        continue

# # Stitch images into composite.
# print('NOW STITCHING...')
# stitchArray(currExp)
# print()

# Crops individual identified rafts.
print('NOW CROPPING TO RAFTS...')
cropRafts(currExp)
print()

# # Performs fluorescence analysis.
# print('NOW PERFORMING FLUORESCENCE ANALYSIS')
# analyzeFluor(currExp, flag_finalTP=0)
# print()

print('PROCESSING COMPLETE.')