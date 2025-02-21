# raftImageAnalysis_main.py - Initializes key parameters for MVX raft image analysis.

from expMeta import ExpMeta
from readMeta import ReadMeta
from stitchArray import stitchArray
from cropRafts import cropRafts
from analyzeFluor_GLR import analyzeFluor

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
        channel_mask = input('Enter channel for mask (na, b, v, g, r, fr): ')
        print()

        # Create experiment metadata object.
        currExp = ExpMeta(dataDir=dataDir,
                          channel_mask=channel_mask)

        print(f'Channel mask: {currExp.channel_mask}')
        print(f'Channels time-series ({len(currExp.channels_tl)} total): {currExp.channels_tl}')
        print(f'Channels endpoint ({len(currExp.channels_end)} total): {currExp.channels_end}')
        print(currExp.reporters_and_stains)
        print()

        # Print out metadata.
        currExp.write_metadata()

        break
    else:
        continue

# Stitch images into composite.
print('NOW STITCHING...')
stitchArray(currExp)
print()

# Crops individual identified rafts.
print('NOW CROPPING TO RAFTS...')
cropRafts(currExp)
print()

# # Performs fluorescence analysis.
# print('NOW PERFORMING FLUORESCENCE ANALYSIS')
# analyzeFluor(currExp)
# print()

print('PROCESSING COMPLETE.')