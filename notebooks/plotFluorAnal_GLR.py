#%%
from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats
import math

#%%
pxSize = 1.0257  # um/px

dataPaths = [Path(r'C:\Users\MVX\Documents\2 - Data\240722_R2GLR_1.8e6_0-48hrBMP4_0%aneu_2_Analysis'),
             Path(r'C:\Users\MVX\Documents\2 - Data\240729_R2GLR_1.8e6_0-48hrBMP4_0%aneu_3_Analysis')
             ]
arrNums = [1, 1]
numTP = [9, 9]
intervals = [6, 6]  # hrs
offset = 0  # hrs
time_lim = [-0.5, 48.5]

#%%
# Features over time

arrays = []
rafts = []
times = []
features = []
ft_types = ['Eccen_Hull', 'FArea_Hull',
            'Den_Green', 'Radius_Green', 'IntMean_Fold_Green', 'IntMean_PerPx_Fold_Green']
values = []

for i in range(len(dataPaths)):
    fluorPath = dataPaths[i] / (f'array{arrNums[i]}_fluorAnalysis_GLR.csv')
    df_fluor = pd.read_csv(str(fluorPath))
    df_fluor = df_fluor[(df_fluor['Missing'] == 0)
                        & (df_fluor['Empty'] == 0)
                        & (df_fluor['Border'] == 0)
                        & (df_fluor['Eccen_Hull'] <= 0.35)
                        ]
    #     df_fluor = df_fluor[(df_fluor['Missing'] == 0)
    #                         & (df_fluor['Empty'] == 0)
    #                         & (df_fluor['Border'] == 0)
    #                        ]
    #     print(len(df_fluor['Raft'].tolist()))
    numRaft = len(set(df_fluor['Raft'].tolist()))
    print(f'Num rafts for Array {i + 1}: {numRaft}')
    for raft in list(set(df_fluor['Raft'].tolist())):
        for tp in range(numTP[i]):
            time = (tp * intervals[i]) + offset
            for ft in ft_types:
                arrays.append(i + 1)
                rafts.append(raft)
                times.append(time)
                features.append(ft)
                val = df_fluor[(df_fluor['Raft'] == raft)
                               & (df_fluor['Time'] == time)][f'{ft}'].item()
                if ft == 'Radius_Green':
                    val *= pxSize
                values.append(val)

df_final = pd.DataFrame({'Array': arrays,
                         'Raft': rafts,
                         'Time': times,
                         'Feature': features,
                         'Value': values
                         })
print(df_final.head(20))

sns.set_style('white')
plt.rcParams['ytick.left'] = True
plt.rcParams['xtick.bottom'] = True
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

sns.scatterplot(ax=axes[0, 0],
                x='Time',
                y='Value',
                data=df_final[df_final['Feature'] == 'Den_Green'],
                hue='Array',
                palette='pastel',
                legend=False)
sns.lineplot(ax=axes[0, 0],
             x='Time',
             y='Value',
             data=df_final[df_final['Feature'] == 'Den_Green'],
             hue='Array',
             linewidth=2,
             legend=False)
axes[0, 0].set_xlabel('')
axes[0, 0].set_xlim(time_lim[0], time_lim[1])
axes[0, 0].set_ylabel(r'Nuclei Density', fontweight='bold')

sns.scatterplot(ax=axes[0, 1],
                x='Time',
                y='Value',
                data=df_final[df_final['Feature'] == 'Radius_Green'],
                hue='Array',
                palette='pastel',
                legend=False)
sns.lineplot(ax=axes[0, 1],
             x='Time',
             y='Value',
             data=df_final[df_final['Feature'] == 'Radius_Green'],
             hue='Array',
             linewidth=2,
             legend=False)
axes[0, 1].set_xlabel('')
axes[0, 1].set_xlim(time_lim[0], time_lim[1])
axes[0, 1].set_ylabel(r'Radius (μm)', fontweight='bold')

sns.scatterplot(ax=axes[1, 0],
                x='Time',
                y='Value',
                data=df_final[df_final['Feature'] == 'IntMean_Fold_Green'],
                hue='Array',
                palette='pastel',
                legend=False)
sns.lineplot(ax=axes[1, 0],
             x='Time',
             y='Value',
             data=df_final[df_final['Feature'] == 'IntMean_Fold_Green'],
             hue='Array',
             linewidth=2,
             legend=False)
axes[1, 0].set_xlabel('Time (h)', fontweight='bold')
axes[1, 0].set_xlim(time_lim[0], time_lim[1])
axes[1, 0].set_ylabel(r'Fold Change in Fluorescence', fontweight='bold')

sns.scatterplot(ax=axes[1, 1],
                x='Time',
                y='Value',
                data=df_final[df_final['Feature'] == 'IntMean_PerPx_Fold_Green'],
                hue='Array',
                palette='pastel',
                legend=False)
sns.lineplot(ax=axes[1, 1],
             x='Time',
             y='Value',
             data=df_final[df_final['Feature'] == 'IntMean_PerPx_Fold_Green'],
             hue='Array',
             linewidth=2,
             legend=False)
axes[1, 1].set_xlabel('Time (h)', fontweight='bold')
axes[1, 1].set_xlim(time_lim[0], time_lim[1])
axes[1, 1].set_ylabel(r'Fold Change in Fluorescence (per px)', fontweight='bold')

sns.set(font_scale=1.25)

#%%
# Features of time (just lineplot with CI)

sns.set_style('white')
plt.rcParams['ytick.left'] = True
plt.rcParams['xtick.bottom'] = True
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

sns.lineplot(ax=axes[0,0],
             x='Time',
             y='Value',
             data = df_final[df_final['Feature'] == 'Den_Green'],
             hue = 'Array',
             linewidth=2,
             legend=False)
axes[0,0].set_xlabel('')
axes[0,0].set_xlim(time_lim[0], time_lim[1])
axes[0,0].set_ylabel(r'Nuclei Density', fontweight='bold')

sns.lineplot(ax=axes[0,1],
             x='Time',
             y='Value',
             data = df_final[df_final['Feature'] == 'Radius_Green'],
             hue = 'Array',
             linewidth=2,
             legend=False)
axes[0,1].set_xlabel('')
axes[0,1].set_xlim(time_lim[0], time_lim[1])
axes[0,1].set_ylabel(r'Radius (μm)', fontweight='bold')

sns.lineplot(ax=axes[1,0],
             x='Time',
             y='Value',
             data = df_final[df_final['Feature'] == 'IntMean_Fold_Green'],
             hue = 'Array',
             linewidth=2,
             legend=False)
axes[1,0].set_xlabel('Time (h)', fontweight='bold')
axes[1,0].set_xlim(time_lim[0], time_lim[1])
axes[1,0].set_ylabel(r'Fold Change in Fluorescence', fontweight='bold')

sns.lineplot(ax=axes[1,1],
             x='Time',
             y='Value',
             data = df_final[df_final['Feature'] == 'IntMean_PerPx_Fold_Green'],
             hue = 'Array',
             linewidth=2,
             legend=False)
axes[1,1].set_xlabel('Time (h)', fontweight='bold')
axes[1,1].set_xlim(time_lim[0], time_lim[1])
axes[1,1].set_ylabel(r'Fold Change in Fluorescence (per px)', fontweight='bold')

sns.set(font_scale=1.25)

#%%
# Distance from center

ringWidth = 10  # px
gasRad = math.ceil((500 / 2 / pxSize) / ringWidth) * ringWidth  # px rounded to nearest multiple of ringWidth
numRings = int(gasRad / ringWidth)

arrays = []
rafts = []
times = []
disFromCens = []
values = []
time_set = []

for i in range(len(dataPaths)):
    fluorPath = dataPaths[i] / (f'array{arrNums[i]}_fluorAnalysis_GLR.csv')
    df_fluor_r = pd.read_csv(str(fluorPath))
    df_fluor_r = df_fluor_r[(df_fluor_r['Missing'] == 0)
                            & (df_fluor_r['Empty'] == 0)
                            & (df_fluor_r['Border'] == 0)
                            & (df_fluor_r['Eccen_Hull'] <= 0.35)
                            ]
    #     print(len(df_fluor_r['Raft'].tolist()))
    time_set = list(set(df_fluor_r['Time'].tolist()))
    time_set.sort()
    for raft in list(set(df_fluor_r['Raft'].tolist())):
        for tp in range(numTP[i]):
            time = (tp * intervals[i]) + offset
            for r in range(numRings):
                arrays.append(i + 1)
                rafts.append(raft)
                times.append(time)

                disFromCen = (ringWidth / 2) + (r * (ringWidth))
                disFromCens.append(disFromCen)
                col_name = f'IntMean_fromCenter_{disFromCen}_Green'

                val = df_fluor_r[(df_fluor_r['Raft'] == raft)
                                 & (df_fluor_r['Time'] == time)][col_name].item()
                val *= pxSize  # um
                values.append(val)

df_final = pd.DataFrame({'Array': arrays,
                         'Raft': rafts,
                         'Time': times,
                         'DisFromCen': disFromCens,
                         'Value': values
                         })
# print(df_final.head(20))

sns.set_style('white')
plt.rcParams['ytick.left'] = True
plt.rcParams['xtick.bottom'] = True
fig, axes = plt.subplots(2, (int(numTP[i] / 2) + 1), figsize=(24, 8))

# print(time_set)

i = 0  # array num
for tp in range(numTP[i]):
    if tp / (int(numTP[i] / 2) + 1) < 1:
        subplot = [0, tp]
    else:
        subplot = [1, tp - (int(numTP[i] / 2) + 1)]
    #     print(subplot)
    sns.scatterplot(ax=axes[subplot[0], subplot[1]],
                    x='DisFromCen',
                    y='Value',
                    data=df_final[df_final['Time'] == time_set[tp]],
                    hue='Array',
                    palette='pastel',
                    legend=False)
    sns.lineplot(ax=axes[subplot[0], subplot[1]],
                 x='DisFromCen',
                 y='Value',
                 data=df_final[df_final['Time'] == time_set[tp]],
                 hue='Array',
                 linewidth=2,
                 legend=False)
    axes[subplot[0], subplot[1]].set_xlabel('Distance From Center (μm)', fontweight='bold')
    axes[subplot[0], subplot[1]].set_xlim(0, gasRad)
    axes[subplot[0], subplot[1]].set_ylabel('')
    #     axes[subplot[0], subplot[1]].set_ylim(0.8, 1.7)
    #     axes[subplot[0], subplot[1]].text(200, 1.4, f'{time_set[tp]} h', fontweight='bold')
    axes[subplot[0], subplot[1]].set_ylim(0.3, 2.5)
    axes[subplot[0], subplot[1]].text(200, 2.3, f'{time_set[tp]} h', fontweight='bold')

if numTP[i] % 2 == 1:
    fig.delaxes(axes[1, numTP[i] - (int(numTP[i] / 2) + 1)])

axes[0, 0].set_ylabel('Normalized Fluorescence', fontweight='bold')
axes[1, 0].set_ylabel('Normalized Fluorescence', fontweight='bold')

sns.set(font_scale=1.5)
fig.tight_layout()

#%%
# Eccentricity

fig, axes = plt.subplots(1, 1, figsize=(4, 4))

sns.set_style('white')
plt.rcParams['ytick.left'] = True
sns.violinplot(y='Value',
              data = df_final[df_final['Feature'] == 'Eccen_Hull'],
              hue = 'Array',
              palette='pastel',
              jitter = 0.3,
              legend=False)
axes.set_ylabel('Hull Eccentricity')

#%%
# FArea

fig, axes = plt.subplots(1, 1, figsize=(4, 4))

sns.set_style('white')
plt.rcParams['ytick.left'] = True
sns.violinplot(y='Value',
              data = df_final[df_final['Feature'] == 'FArea_Hull'],
              hue = 'Array',
              palette='pastel',
              jitter = 0.3,
              legend=False)
axes.set_ylabel('Hull FArea')