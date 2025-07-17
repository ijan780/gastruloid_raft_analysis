# expMeta.py - Stores experiment metadata for Python functions to access.

from pathlib import Path
import os
import pandas as pd
from parse import parse
import ast
import math

class ExpMeta(object):
    def __init__(self,
                 dataDir: str,
                 channel_mask=None,
                 channels_tl=[],
                 endDirSame=True,
                 load=False
                 ):
        # Fairly consistent variables
        self.arrDim = [800, 100, 23]  # [side dimensions, gap spacing, # rafts in single row]

        # Pixel_Sizes = [5.1587 4.0625 3.2244 2.6558 2.068 1.6535 1.3163 1.0257 0.8203 0.6522 0.5247]
        self.dict_pxSize_2x = {1.26: 5.1587,
                               1.6: 4.0625,
                               2: 3.2244,
                               2.5: 2.6558,
                               3.2: 2.068,
                               4: 1.6535,
                               5: 1.3163,
                               6.4: 1.0257,
                               8: 0.8203,
                               10: 0.6522,
                               12.6: 0.5247
                               }  # pxSizes (um/px) for 2x objective across magnifications
        self.mag = 4
        self.pxSize = self.dict_pxSize_2x[self.mag]
        self.channels_abb = {'na': None,
                             'all': 'all',
                             'BF': 'BF',
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
                                     'tl_Red': 'SOX17',
                                     'end_BF': 'BF',
                                     }

        # Variables that change for each experiment
        if load is False:
            self.set_dataDir(dataDir)
            self.analDir = Path(f'{self.dataDir}_Analysis')
            self.analDir.mkdir(exist_ok=True)

            self.pattern = '{timePoint}_{channel}.tif'
            firstFolder = self.find_firstFolder(dataDir)
            self.set_numArrs()
            self.set_time(firstFolder)

            self.channel_mask = self.channels_abb[channel_mask]

            while self.channel_mask is not None:
                endDirSame = input('Is the mask folder the same as the data folder? (y/n): ')
                if endDirSame.lower() == 'y':
                    endDirSame = True
                    break
                elif endDirSame.lower() == 'n':
                    endDirSame = False
                    break
                else:
                    continue
            self.set_endDir(endDirSame)

            channels_main = self.find_channels(firstFolder) # channels in dataDir
            if (self.numTimePoints > 1):
                channels_tl = input('Enter reporter channels separated by spaces (all, b, v, g, r, fr): ')
                channels_tl = channels_tl.split()
            self.set_channels(channels_main, channels_tl, endDirSame)
            print(f'Channels ({len(self.channels)} total): {self.channels}')
            print()

            if self.channels_end:
                self.set_endStains()

        else:
            self.read_metadata(dataDir)

        # Variables that modify behavior of image analysis
        self.tileScale = 0.1

    # Ensure that directory path input is submitted.
    def set_dataDir(self, dataDir):
        if dataDir == '':
            raise ValueError('You forgot to enter a directory path!')
        self.dataDir = Path(dataDir)

    def set_endDir(self, endDirSame):
        if (endDirSame) or (self.channel_mask == None):
            self.endDir = self.dataDir
        else:
            self.endDir = Path(f'{self.dataDir}_End')

    def find_firstFolder(self, dir):
        return Path(dir) / '1' / '1x1'  # folder for first subsection.

    def set_numArrs(self):
        self.numArrs = len(os.listdir(self.dataDir)) - 1  # account for "data" subfolder
        print(f'Number of arrays: {self.numArrs}')
    
    def set_time(self, firstFolder):
        lastImg = os.listdir(firstFolder)[-1]  # last image in folder for first subsection
        patternResult = parse(self.pattern, lastImg)
        self.numTimePoints = int(patternResult['timePoint']) + 1
        print(f'Number of time points: {self.numTimePoints}')

        if self.numTimePoints == 1:
            interval = 99.0
            timeOffset = 0.0
        else:
            interval = float(input('Enter duration of time interval (hr): '))
            timeOffset = float(input('Enter time offset (hr): '))

        self.set_interval(interval)
        self.set_timeOffset(timeOffset)

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
        if self.numTimePoints == 1:
            self.channels_end = channels_main
            self.channels_tl = []
        elif 'all' in channels_tl:
            self.channels_end = []
            self.channels_tl = channels_main
        else:
            self.channels_tl = ['BF'] + [self.channels_abb[ch] for ch in channels_tl]
            if endDirSame:
                channels_end = [ch for ch in channels_main if ch not in self.channels_tl]
            else:
                firstFolder_end = self.find_firstFolder(self.endDir)
                channels_end = self.find_channels(firstFolder_end)
                channels_end = channels_end[1:] # exclude BF if endpoint images in separate directory
            self.channels_end = [self.channel_mask] + [ch for ch in channels_end if ch != self.channel_mask]
        self.channels = self.channels_tl + self.channels_end

    def set_endStains(self):
        if self.channel_mask is not None:
            self.reporters_and_stains[f'end_{self.channel_mask}'] = 'Hull'
        for ch in [ch for ch in self.channels_end if (ch not in ['BF', self.channel_mask])]:
            stain = input(f'Stain that the following endpoint channel [{ch}] refers to: ')
            self.reporters_and_stains[f'end_{ch}'] = stain

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

    def read_metadata(self, dataDir):
        metaDir = Path(f'{dataDir}_Analysis') / 'Experiment Metadata.csv'
        df_meta = pd.read_csv(metaDir)
        dict_meta = df_meta.set_index('Properties')['Values'].to_dict()

        self.dataDir = Path(dataDir)
        self.endDir = Path(dict_meta['End Dir'])
        self.endDirSame = (dict_meta['Data Dir'] == dict_meta['End Dir'])
        self.analDir = Path(dict_meta['Analysis Dir'])
        self.interval = float(dict_meta['Time Interval (hrs)'])
        self.timeOffset = float(dict_meta['Time Offset (hrs)'])
        self.numTimePoints = int(dict_meta['Number of Time Points'])
        self.numArrs = int(dict_meta['Number of Arrays'])
        self.channels = ast.literal_eval(dict_meta['Channels (total)'])
        self.reporters_and_stains = ast.literal_eval(dict_meta['Reporters & Stains'])
        if math.isnan(dict_meta['Channel (mask)']):
            self.channel_mask = None
        else:
            self.channel_mask = dict_meta['Channel (mask)']
        self.channels_tl = ast.literal_eval(dict_meta['Channels (tl)'])
        self.channels_end = ast.literal_eval(dict_meta['Channels (end)'])
        self.mag = float(dict_meta['Magnification'])
        self.pxSize = float(dict_meta['Pixel Size'])
        self.tileScale = float(dict_meta['Scale for Stitched Composites'])