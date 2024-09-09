# raftImageAnalysis_main.py - Initializes key parameters for MVX image analysis. Currently stitches subsections
# and crops images to individual identified rafts.

from pathlib import Path
import os
from parse import parse

from expMeta import ExpMeta
from stitchArray import stitchArray
from cropRafts import cropRafts
from analyzeFluor_CalcHoechst import analyzeFluor

# Request user input for data folder absolute path and time intervals.
dataPath = input('Enter data folder absolute path: ')
# dataPath = Path(r'C:\Users\ianja\OneDrive - UW\Allbritton Lab\MVX System\Image Analysis Pipeline\220815_ImagingTest')
interval = input('Enter duration of time interval (hr): ')
interval = float(interval)
currExp = ExpMeta(dataPath, interval)

# Determine number of arrays.
currExp.numArrs = len(os.listdir(dataPath)) - 1 # account for "data" subfolder
print(f'Number of arrays: {currExp.numArrs}')

# Folder for first subsection.
randomImgFolder = Path(currExp.dataPath / '1' / '1x1')

# Determine number of time points.
pattern = '{timePoint}_{channel}.tif'
lastImg = os.listdir(randomImgFolder)[-1]   # last image in folder for first subsection
patternResult = parse(pattern, lastImg)
# currExp.numTimePoints = 1
currExp.numTimePoints = int(patternResult['timePoint']) + 1
print(f'Number of time points: {currExp.numTimePoints}')
if currExp.numTimePoints == 1:
    currExp.set_interval(99.0)

# List all channels.
for filename in os.listdir(randomImgFolder):
    patternResult = parse(pattern, filename)
    if patternResult['channel'] not in currExp.channels:
        currExp.channels.append(patternResult['channel'])
print(f'Channels ({len(currExp.channels)} total): {currExp.channels}')
print('')

# print('The new experiment metadata object is properly created.')

# Stitch images into composite.
print('NOW STITCHING...')
stitchArray(currExp)
print('')

# Crops individual identified rafts.
print('NOW CROPPING TO RAFTS...')
cropRafts(currExp)
print('')

# Print out metadata.
currExp.write_metadata()

# # Performs fluorescence analysis.
# print('NOW PERFORMING FLUORESCENCE ANALYSIS')
# analyzeFluor(currExp, flag_finalTP=0)
# print('')

print('PROCESSING COMPLETE.')