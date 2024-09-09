# expMeta.py - Stores experiment metadata for Python functions to access.

from pathlib import Path
import pandas as pd

class ExpMeta(object):
    def __init__(self, dir = '', timeInt = 99.0, timeOffset = 0):
        # Fairly consistent variables
        self.arrDim = [800, 100, 23]  # [side dimensions, gap spacing, # rafts in single row]
        # self.mag = 2.5
        # self.pxSize = 1.3163  # um/px (for 2.5X magnification)
        self.mag = 3.2
        self.pxSize = 1.0257  # um/px (for 3.2X magnification)
        # self.pattSub = '{timePoint}_{channel}.tif'
        # self.pattCropRaft = 'array{arr}_timepoint{timePoint}_{channel}_raft{raft}.tif'

        # Variables that change for each experiment
        self.set_dataPath(dir)
        self.set_interval(timeInt)
        self.set_timeOffset(timeOffset)
        self.numArrs = 1
        self.numTimePoints = 1
        self.channels = ['BF']
        # self.channels = ['BF'] # no flat-field correction

        # Variables that modify behavior of image analysis
        self.tileScale = 0.1

    # Ensure that directory path input is submitted.
    def set_dataPath(self, dir):
        if dir == '':
            raise ValueError('You forgot to enter a directory path!')
        self.dataPath = Path(dir)

    # Ensure that time interval input is positive.
    def set_interval(self, timeInt):
        if timeInt < 0:
            raise ValueError('Time interval has to be positive!')
        self.interval = timeInt

    def set_timeOffset(self, timeOffset):
        if timeOffset < 0:
            raise ValueError('Time offset has to be positive!')
        self.timeOffset = timeOffset

    def write_metadata(self):
        totalDuration = (self.interval * (self.numTimePoints - 1))
        properties = ['Data Path',
                      'Time Interval (hrs)',
                      'Total Duration (hrs)',
                      'Number of Time Points',
                      'Time Offset (hrs)',
                      'Number of Array(s)',
                      'Number of Channels',
                      'Channels',
                      'Magnification',
                      'Pixel Size',
                      'Scale for Stitched Composites']
        vals = [self.dataPath,
                self.interval,
                "{:.2f}".format(totalDuration),
                self.numTimePoints,
                self.timeOffset,
                self.numArrs,
                len(self.channels),
                self.channels,
                self.mag,
                self.pxSize,
                self.tileScale]

        outPath = Path(self.dataPath.parent.absolute() /
                       f'{self.dataPath}_Analysis'/
                       'Experiment Metadata.csv')
        df_meta = pd.DataFrame({"Properties": properties,
                                "Values": vals})
        df_meta.to_csv(outPath, header=True, index=False)

        # f = open(outPath.__str__(), 'w')
        # for i in range(0, len(properties)):
        #     f.write(f'{properties[i]}{str(metadata[i])}\n')
        # f.close()

        print('SAVED EXPERIMENT METADATA AS CSV.\n')