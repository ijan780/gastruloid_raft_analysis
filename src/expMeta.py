# expMeta.py - Stores experiment metadata for Python functions to access.

from pathlib import Path
import pandas as pd

class ExpMeta(object):
    def __init__(self,
                 dataDir: str,
                 channels: list,
                 channel_mask: str,
                 channels_tl: list,
                 timeInt: float,
                 maskDirSame = True,
                 timeOffset: float = 0.0,
                 numArrs: int = 1,
                 numTimePoints: int = 1,
                 ):
        # Fairly consistent variables
        self.arrDim = [800, 100, 23]  # [side dimensions, gap spacing, # rafts in single row]
        # self.mag = 2.5
        # self.pxSize = 1.3163  # um/px (for 2.5X magnification)
        self.mag = 3.2
        self.pxSize = 1.0257  # um/px (for 3.2X magnification)

        # Variables that change for each experiment
        self.set_dataDir(dataDir)
        self.set_maskDir(dataDir, maskDirSame)
        self.analDir = Path(f'{self.dataDir}_Analysis')
        self.analDir.mkdir(exist_ok=True)

        self.numArrs = numArrs
        self.numTimePoints = numTimePoints

        self.set_interval(timeInt)
        self.set_timeOffset(timeOffset)

        self.channels = channels
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
        self.channel_mask = self.channels_abb[channel_mask]
        self.set_tl_end(channels_tl)

        # Variables that modify behavior of image analysis
        self.tileScale = 0.1

    # Ensure that directory path input is submitted.
    def set_dataDir(self, dataDir):
        if dataDir == '':
            raise ValueError('You forgot to enter a directory path!')
        self.dataDir = Path(dataDir)

    def set_maskDir(self, dataDir, maskDirSame):
        if maskDirSame:
            self.maskDir = self.dataPath
        else:
            self.maskDir = Path(f'{dataDir}_Calcein')

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

    def set_tl_end(self, channels_tl):
        if (self.numTimePoints == 1) or channels_tl[0] == 'none':
            self.channels_tl = []
            self.channels_end = self.channels
        elif channels_tl[0] == 'all':
            self.channels_tl = self.channels
            self.channels_end = []
        else:
            self.channels_tl = [self.channels_abb[ch] for ch in channels_tl]
            self.channels_end = [ch for ch in self.channels if ch not in self.channels_tl]

    def write_metadata(self):
        totalDuration = (self.interval * (self.numTimePoints - 1))
        properties = ['Data Dir',
                      'Mask Dir',
                      'Analysis Dir',
                      'Time Interval (hrs)',
                      'Total Duration (hrs)',
                      'Time Offset (hrs)',
                      'Number of Time Points',
                      'Number of Arrays',
                      'Number of Channels',
                      'Channels (total)',
                      'Channel (mask)',
                      'Channels (tl)',
                      'Channels (end)',
                      'Magnification',
                      'Pixel Size',
                      'Scale for Stitched Composites']
        vals = [self.dataDir,
                self.maskDir,
                self.analDir,
                self.interval,
                "{:.2f}".format(totalDuration),
                self.timeOffset,
                self.numTimePoints,
                self.numArrs,
                len(self.channels),
                self.channels,
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