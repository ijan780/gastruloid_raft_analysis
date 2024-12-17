# fixRadius.py - Divides "radius" by 2 in all fluorAnalysis csv files.

from pathlib import Path
import os
import pandas as pd
import numpy as np

#%%
# dir = r'C:\Users\MVX\Documents\2 - Data\240716_R2GLR_1.8e6_21-48hrBMP4_0%aneu_1_Analysis\array1_FluorAnalysis_GLR.csv'
# channels = ['Hull', 'Green']
# outPath = str(dir)
# df = pd.read_csv(outPath)
# for ch in channels:
#     radius = df[f'Radius_{ch}'].tolist()
#     radius = [(r / 2) for r in radius]
#     df[f'Radius_{ch}'] = radius
# df.to_csv(outPath, header=True, index=False)

# dir = r'C:\Users\MVX\Documents\2 - Data\Fixed'
# channels = ['Blue', 'Green', 'Red', 'FarRed']
# for subdir, dirs, files in os.walk(dir):
#     for file in files:
#         if 'FluorAnalysis_Fixed.csv' in file:
#             print(file)
#             outPath = str(Path(subdir) / file)
#             df = pd.read_csv(outPath)
#
#             for ch in channels:
#                 radius = df[f'Radius_{ch}'].tolist()
#                 radius = [(r/2) for r in radius]
#                 df[f'Radius_{ch}'] = radius
#             df.to_csv(outPath, header=True, index=False)

# for subdir, dirs, files in os.walk(dir):
#     for file in files:
#         if 'FluorAnalysis_CalcHoechst.csv' in file:
#             print(file)
#             outPath = str(Path(subdir) / file)
#             df = pd.read_csv(outPath)
#             radius = df['Radius'].tolist()
#             radius = [(r/2) for r in radius]
#             df['Radius'] = radius
#             df.to_csv(outPath, header=True, index=False)

#%%
dataDir = Path(r'C:\Users\MVX\Documents\2 - Data\Analysis\Live')
for subdir in dataDir.iterdir():
    for file in subdir.iterdir():
        if 'FluorAnalysis_CalcHoechst.csv' in str(file):
            print(file)

            df = pd.read_csv(file)
            # df.drop('Unnamed: 0', axis=1, inplace=True)
            radius = df['Radius'].tolist()
            radius = [(r / 2) for r in radius]
            df['Radius'] = radius
            df.to_csv(file, index=False)

#%%
dataDir = Path(r'C:\Users\MVX\OneDrive - UW\Allbritton Lab\Manuscripts\Manuscript 1 - Technical\Data\Eu & Aneu Gastruloid Fluor Analysis')
for file in dataDir.iterdir():
    if 'FluorAnalysis' in str(file):
        print(file)

        df = pd.read_csv(file)
        radius = df['Radius'].tolist()
        radius = [(r / 2) for r in radius]
        df['Radius'] = radius
        df.to_csv(file, index=False)