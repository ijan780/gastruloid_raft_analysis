# readMeta.py - loads from metadata csv file.

from pathlib import Path
import pandas as pd
import ast
from src.expMeta import ExpMeta

def ReadMeta(dataDir):
    metaDir = Path(f'{dataDir}_Analysis') / 'Experiment Metadata.csv'
    df_meta = pd.read_csv(metaDir)
    dict_meta = df_meta.set_index('Properties')['Values'].to_dict()

    dataDir = dict_meta['Data Dir']
    channel_mask = dict_meta['Channel (mask)']
    channels_tl = ast.literal_eval(dict_meta['Channels (tl)'])
    interval = float(dict_meta['Time Interval (hrs)'])
    endDirSame = (dict_meta['Data Dir'] == dict_meta['End Dir'])
    timeOffset = float(dict_meta['Time Offset (hrs)'])

    # Create experiment metadata object.
    currExp = ExpMeta(dataDir=dataDir,
                      channel_mask=channel_mask,
                      channels_tl=channels_tl,
                      timeInt=interval,
                      endDirSame=endDirSame,
                      timeOffset=timeOffset)
    currExp.reporters_and_stains = ast.literal_eval(dict_meta['Reporters & Stains'])
    print()

    return currExp
