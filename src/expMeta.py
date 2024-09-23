# expMeta.py - Stores experiment metadata for Python functions to access.

from pathlib import Path
import os
import pandas as pd
from parse import parse

class ExpMeta(object):
    def __init__(self,
                 dataDir: str,
                 channel_mask: str,
                 channels_tl: list,
                 timeInt: float,
                 endDirSame = True,
                 timeOffset: float = 0.0,
                 ):
        # Fairly consistent variables
        self.arrDim = [800, 100, 23]  # [side dimensions, gap spacing, # rafts in single row]
        # self.mag = 2.5
        # self.pxSize = 1.3163  # um/px (for 2.5X magnification)
        self.mag = 3.2
        self.pxSize = 1.0257  # um/px (for 3.2X magnification)

        # Variables that change for each experiment
        self.set_dataDir(dataDir)
        self.set_endDir(endDirSame)
        self.analDir = Path(f'{self.dataDir}_Analysis')
        self.analDir.mkdir(exist_ok=True)

        self.pattern = '{timePoint}_{channel}.tif'
        firstFolder = self.find_firstFolder(dataDir)
        self.set_numArrs()
        self.set_numTimePoints(firstFolder)

        self.set_interval(timeInt)
        self.set_timeOffset(timeOffset)

        self.channels_abb = {'BF': 'BF',
                             'bf': 'BF',
                             'Blue': 'Blue',
                             'b': 'Blue',
                             'Violet': 'Violet',
                             'v': 'Violet',
                             'Green': 'Green',
                             'g': 'Green',
                             'Red': 'Red',
                             'r': 'Red',
                             'FarRed': 'FarRed',
                             'fr': 'FarRed'}
        self.reporters_and_stains = {'tl_BF': 'BF',
                                     'tl_Violet': 'BRA',
                                     'tl_Green': 'SOX2',
                                     'tl_Red': 'SOX17'}
        self.channel_mask = self.channels_abb[channel_mask]
        channels_main = self.find_channels(firstFolder) # channels in dataDir
        self.set_channels(channels_main, channels_tl, endDirSame)
        print(f'Channels ({len(self.channels)} total): {self.channels}')

        # Variables that modify behavior of image analysis
        self.tileScale = 0.1

    # Ensure that directory path input is submitted.
    def set_dataDir(self, dataDir):
        if dataDir == '':
            raise ValueError('You forgot to enter a directory path!')
        self.dataDir = Path(dataDir)

    def set_endDir(self, endDirSame):
        if endDirSame:
            self.endDir = self.dataDir
        else:
            self.endDir = Path(f'{self.dataDir}_Calcein') # TODO edit to _End

    def find_firstFolder(self, dir):
        return Path(dir) / '1' / '1x1'  # folder for first subsection.

    def set_numArrs(self):
        self.numArrs = len(os.listdir(self.dataDir)) - 1  # account for "data" subfolder
        print(f'Number of arrays: {self.numArrs}')
    
    def set_numTimePoints(self, firstFolder):
        lastImg = os.listdir(firstFolder)[-1]  # last image in folder for first subsection
        patternResult = parse(self.pattern, lastImg)
        self.numTimePoints = int(patternResult['timePoint']) + 1
        print(f'Number of time points: {self.numTimePoints}')

    # Ensure that time interval input is positive.
    def set_interval(self, timeInt):
        if timeInt < 0:
            raise ValueError('Time interval has to be positive!')
        elif self.numTimePoints == 1:
            self.interval = 99.0
        else:
            self.interval = float(timeInt)

    def set_timeOffset(self, timeOffset):
        if timeOffset < 0:
            raise ValueError('Time offset has to be positive!')
        self.timeOffset = float(timeOffset)

    def find_channels(self, firstFolder):
        channels = ['BF']
        for filename in os.listdir(firstFolder):
            patternResult = parse(self.pattern, filename)
            if patternResult['channel'] not in channels:
                channels.append(patternResult['channel'])
        return channels

    def set_channels(self, channels_main, channels_tl, endDirSame):
        # channels_tl refers to reporter channels + BF.
        if (self.numTimePoints == 1) or channels_tl[0] == 'none':
            self.channels_tl = []
            self.channels_end = channels_main
        elif channels_tl[0] == 'all':
            self.channels_tl = channels_main
            self.channels_end = []
        else:
            self.channels_tl = ['BF'] + [self.channels_abb[ch] for ch in channels_tl]
            if endDirSame:
                self.channels_end = [ch for ch in channels_main if ch not in self.channels_tl]
            else:
                firstFolder_end = self.find_firstFolder(self.endDir)
                channels_end = self.find_channels(firstFolder_end)
                self.channels_end = channels_end[1:] # exclude BF if endpoint images in separate directory
        self.channels = self.channels_tl + self.channels_end

    def set_endStains(self):
        self.reporters_and_stains[f'end_{self.channel_mask}'] = 'Hull'
        for ch in [ch for ch in self.channels_end if ch != self.channel_mask]:
            stain = input(f'Stain that the following endpoint channel [{ch}] refers to: ')
            self.reporters_and_stains[f'end_{ch}'] = stain
        print(self.reporters_and_stains)

    def write_metadata(self):
        totalDuration = (self.interval * (self.numTimePoints - 1))
        properties = ['Data Dir',
                      'End Dir',
                      'Analysis Dir',
                      'Time Interval (hrs)',
                      'Total Duration (hrs)',
                      'Time Offset (hrs)',
                      'Number of Time Points',
                      'Number of Arrays',
                      'Number of Channels',
                      'Channels (total)',
                      'Reporters & Stains',
                      'Channel (mask)',
                      'Channels (tl)',
                      'Channels (end)',
                      'Magnification',
                      'Pixel Size',
                      'Scale for Stitched Composites']
        vals = [self.dataDir,
                self.endDir,
                self.analDir,
                self.interval,
                "{:.2f}".format(totalDuration),
                self.timeOffset,
                self.numTimePoints,
                self.numArrs,
                len(self.channels),
                self.channels,
                self.reporters_and_stains,
                self.channel_mask,
                self.channels_tl,
                self.channels_end,
                self.mag,
                self.pxSize,
                self.tileScale]

        metaDir = self.analDir / 'Experiment Metadata.csv'
        df_meta = pd.DataFrame({"Properties": properties,
                                "Values": vals})
        df_meta.to_csv(metaDir, header=True, index=False)

        # f = open(outPath.__str__(), 'w')
        # for i in range(0, len(properties)):
        #     f.write(f'{properties[i]}{str(metadata[i])}\n')
        # f.close()

        print('SAVED EXPERIMENT METADATA AS CSV.\n')