#%%
import os
from pathlib import Path
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import scipy.stats as stats
from statsmodels.stats.weightstats import DescrStatsW
import statistics
import math

#%%
dataDir = Path(r'C:\Users\MVX\OneDrive - UW\Allbritton Lab\Manuscripts\Manuscript 1 - Technical\Data\Fixed Gastruloid Fluor Analysis')
os.chdir(dataDir)
arrays = [_ for _ in range(1, 4)]
pxSize = 1.3163  # um/px

arrays_col = []
rafts_col = []

chs = ['Blue', 'Green', 'Red', 'FarRed']
fts = ['ThreshVal', 'FArea', 'Dens', 'Eccen', 'Radius']

total_fts = []
for ch in chs:
    for ft in fts:
        total_fts.append(f'{ft}_{ch}')

var_names = []
for idx, ft in enumerate(fts):
    var_names_ch = [f'{ft}_{ch}' for ch in chs]
    var_names.append(var_names_ch)

#%%
df_fluors = pd.DataFrame()

for array in arrays:
    df_fluor = pd.read_csv(f'array{array}_FluorAnalysis_Fixed.csv')
    df_fluor_filter = df_fluor[(df_fluor['Missing'] == 0) &
                               (df_fluor['Empty'] == 0) &
                               (df_fluor['Border'] == 0) &
                               (df_fluor['Lo_Int'] == 0)]
    df_fluor_filter['Array'] = [array] * df_fluor_filter.shape[0]

    for ft in var_names[-1]:
        df_fluor_filter[ft] = [(radius * pxSize) for radius in df_fluor_filter[ft].tolist()]

    df_fluor_melt = df_fluor_filter.melt(id_vars=['Array', 'Raft'],
                                         value_vars=total_fts
                                         )
    df_fluors = pd.concat([df_fluors, df_fluor_melt],
                          axis=0,
                          ignore_index=True)

#%%
arrs_col2 = []
var_names_col2 = []
means_col2 = []
stdevs_col2 = []

for arr in arrays:
    for var_name in var_names:
        for ft in var_name:
            arrs_col2.append(arr)
            var_names_col2.append(ft)

            df_sub = df_fluors[(df_fluors['Array'] == arr) &
                               (df_fluors['variable'] == ft)]
            mean = np.mean(df_sub['value'])
            stdev = np.std(df_sub['value'])

            means_col2.append(mean)
            stdevs_col2.append(stdev)

df_means = pd.DataFrame({'Array': arrs_col2,
                         'Feat': var_names_col2,
                         'Mean': means_col2,
                         'Stdev': stdevs_col2})

#%%
var_names_plot = var_names[1:]

plt.rcdefaults()
sns.set_theme(style='white', font_scale=1.25)
plt.rcParams['ytick.left'] = True
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for idx, var_name in enumerate(var_names_plot):
    row = int(idx / 2)
    col = idx % 2

    df_sub = df_fluors[df_fluors['variable'].isin(var_name)]
    df_sub_mean = df_means[df_means['Feat'].isin(var_name)]

    sns.stripplot(ax=axes[row, col],
                  x='variable',
                  y='value',
                  data=df_sub,
                  hue='Array',
                  size=3,
                  alpha=0.5,
                  palette='colorblind',
                  jitter=0.3,
                  legend=False)
    sns.swarmplot(ax=axes[row, col],
                  x='Feat',
                  y='Mean',
                  data=df_sub_mean,
                  hue='Array',
                  size=15,
                  palette='colorblind',
                  edgecolor='k',
                  linewidth=2,
                  legend=False)
    axes[row, col].set_xlabel('')
    axes[row, col].set_xticks([0, 1, 2, 3], ['Hoechst', 'SOX2', 'BRA', 'CDX2'], fontweight='bold')

axes[0, 0].set_ylabel(r'Fluorescence Area ($\mathregular{10^5}$ $\mathregular{px^2}$)', fontweight='bold')
axes[0, 0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
axes[0, 0].yaxis.get_offset_text().set_visible(False)

axes[0, 1].set_ylabel('Density', fontweight='bold')

axes[1, 0].set_ylabel('Eccentricity', fontweight='bold')

axes[1, 1].set_ylabel('Radius (μm)', fontweight='bold')

plt.tight_layout()
plt.show()

#%%
dataPaths = [Path(r'C:\Users\MVX\Documents\2 - Data\231019_RUES2_1.8e6_48hrBMP4_0%aneu_1_Analysis'),
             Path(r'C:\Users\MVX\Documents\2 - Data\231019_RUES2_1.8e6_48hrBMP4_0%aneu_2_Analysis'),
             Path(r'C:\Users\MVX\Documents\2 - Data\231019_RUES2_1.8e6_48hrBMP4_0%aneu_3_Analysis')]
arrNums = ['1', '1', '1']

arrays = []
rafts = []
channels = []
ch_types = ['Blue', 'Green', 'Red', 'FarRed']
features = []
ft_types = ['FArea', 'ThreshVal', 'Dens', 'Eccen', 'Radius']
values = []

for i in range(len(dataPaths)):
    fluorPath = dataPaths[i] / (f'array{arrNums[i]}_fluorAnalysis_Fixed.csv')
    df_fluor = pd.read_csv(str(fluorPath))
    df_fluor = df_fluor[(df_fluor['Missing'] == 0)
                        & (df_fluor['Empty'] == 0)
                        & (df_fluor['Border'] == 0)
                        ]
    #     print(len(df_fluor['Raft'].tolist()))
    for raft in df_fluor['Raft'].tolist():
        for ch in ch_types:
            for ft in ft_types:
                arrays.append(i + 1)
                rafts.append(raft)
                channels.append(ch)
                features.append(ft)
                col_name = f'{ft}_{ch}'
                val = df_fluor[df_fluor['Raft'] == raft][col_name].item()
                values.append(val)

df_final = pd.DataFrame({'Array': arrays,
                         'Raft': rafts,
                         'Channel': channels,
                         'Feature': features,
                         'Value': values
                         })
min_thresh1 = []
for ch in ch_types:
    df_sub = df_final[(df_final['Channel'] == ch)
                      & (df_final['Feature'] == 'ThreshVal')
                      & (df_final['Array'] != 3)]
    min_thresh1.append(min(df_sub['Value'].tolist()))
min_thresh2 = []
for ch in ch_types:
    df_sub = df_final[(df_final['Channel'] == ch)
                      & (df_final['Feature'] == 'ThreshVal')
                      & (df_final['Array'] == 3)]
    min_thresh2.append(min(df_sub['Value'].tolist()))
print(f'Min threshold values for Arrays 1 & 2: {min_thresh1}')
print(f'Min threshold values for Array 3: {min_thresh2}')

fig = plt.figure(figsize=(4.5, 4))
sns.set_style('white')
plt.rcParams['ytick.left'] = True

sns.stripplot(x='Channel',
              y='Value',
              data=df_final[df_final['Feature'] == 'ThreshVal'],
              hue='Array',
              size=3,
              palette='deep',
              jitter=0.3,
              legend=False)
plt.xlabel('')
plt.xticks([0, 1, 2, 3], ['Hoechst', 'SOX2', 'BRA', 'CDX2'], fontweight='bold')
plt.ylabel('Threshold Value', fontweight='bold')
plt.ylim(0, 400)

sns.set(font_scale=1.25)

#%%
exclude_list = [52, 8, 175, 48, 12, 61, 72, 15, 0, 42, 4, 86]
numArr = len(exclude_list)
pass_list = []
for ex in exclude_list:
    pass_list.append(529 - ex)
# print(sum(pass_list),sum(exclude_list))
missing_list = [4, 5, 10, 6, 0, 0, 0, 0, 0, 2, 2, 0]
empty_list = [13, 3, 1, 0, 8, 1, 3, 8, 0, 37, 1, 72]
border_list = [36, 0, 164, 42, 4, 60, 69, 7, 0, 3, 1, 14]

arrays = []
pers = []
types = []

for i in range(numArr):
    arrays.append(i + 1)
    pers.append(pass_list[i] / 529)
    types.append('Pass')
    arrays.append(i + 1)
    pers.append(exclude_list[i] / 529)
    types.append('Exclude')
    arrays.append(i + 1)
    if exclude_list[i] == 0:
        pers.append(-1)
        types.append('Missing')
        arrays.append(i + 1)
        pers.append(-1)
        types.append('Empty')
        arrays.append(i + 1)
        pers.append(-1)
        types.append('Border')
    else:
        pers.append(missing_list[i] / exclude_list[i])
        types.append('Missing')
        arrays.append(i + 1)
        pers.append(empty_list[i] / exclude_list[i])
        types.append('Empty')
        arrays.append(i + 1)
        pers.append(border_list[i] / exclude_list[i])
        types.append('Border')

# Set 6 EU has 0 excluded rafts
exclude_list2 = exclude_list
del exclude_list2[8]
numArr_adj = list(range(numArr))
del numArr_adj[8]

df = pd.DataFrame({'Array': arrays,
                   'Type': types,
                   'Per': pers})
pass_list_per = df[df['Type']=='Pass']['Per'].tolist()
exclude_list_per = df[df['Type']=='Exclude']['Per'].tolist()

passex_mean = [statistics.mean(pass_list_per),
               statistics.mean(exclude_list_per)]
passex_stdev = [statistics.stdev(pass_list_per),
                statistics.stdev(exclude_list_per)]
print(f'Pass vs Exclude Means: {passex_mean}')
print(f'Pass vs Exclude Stdev: {passex_stdev}')
print()

# Automated weighted average
missing_list2 = missing_list
empty_list2 = empty_list
border_list2 = border_list
del missing_list2[8]
del empty_list2[8]
del border_list2[8]
for i in range(numArr - 1):
    missing_list2[i] = missing_list2[i] / exclude_list2[i]
    empty_list2[i] = empty_list2[i] / exclude_list2[i]
    border_list2[i] = border_list2[i] / exclude_list2[i]
missing_w = DescrStatsW(missing_list2, weights=exclude_list2, ddof=0)
empty_w = DescrStatsW(empty_list2, weights=exclude_list2, ddof=0)
border_w = DescrStatsW(border_list2, weights=exclude_list2, ddof=0)
ex_mean = [missing_w.mean, empty_w.mean, border_w.mean]
ex_se = [missing_w.std_mean, empty_w.std_mean, border_w.std_mean]
print(f'Excluded Type Means: {ex_mean}')
print(sum(ex_mean))
print(f'Excluded Type SE: {ex_se}')

# # Manual weighted average
# missing = 0
# empty = 0
# border = 0
# for i in numArr_adj:
#     print(i)
#     missing += missing_list[i]
#     empty += empty_list[i]
#     border += border_list[i]
# ex_mean = [missing, empty, border]
# for i in range(len(ex_mean)):
#     ex_mean[i] = ex_mean[i] / sum(exclude_list2)
# print(ex_mean)

fig = plt.figure(figsize=(4.5, 4))
sns.set_style('white')
plt.rcParams['ytick.left'] = True

# Pass vs exclude
df_passex = df[(df['Type'] == 'Pass') | (df['Type'] == 'Exclude')]
sns.swarmplot(x='Type',
              y='Per',
              data=df_passex,
              hue='Array',
              size=10,
              palette='pastel',
              legend=False,
              zorder=1)
for i in range(2):
    plt.errorbar(x=i,
                 y=passex_mean[i],
                 yerr=passex_stdev[i],
                 color='black',
                 capsize=10,
                 capthick=1.5,
                 marker='_',
                 markersize=40,
                 markeredgewidth=1.5,
                 zorder=2,
                 linewidth=1.5)
plt.xticks([0, 1],['Pass', 'Exclude'], fontweight='bold')
plt.xlim(-0.5, 1.5)
plt.xlabel('')
plt.ylabel('Proportion of All Rafts', fontweight='bold')

sns.set(font_scale=1.25)
# fig.tight_layout()

#%%
fig = plt.figure(figsize=(4.5, 4))
sns.set_style('white')
plt.rcParams['ytick.left'] = True

# Excluded rafts
df_ex = df[((df['Type'] == 'Missing') | (df['Type'] == 'Empty') | (df['Type'] == 'Border')) &
           (df['Per'] > -1)
          ]
sns.swarmplot(x='Type',
              y='Per',
              data= df_ex,
              hue='Array',
              size=10,
              palette='pastel',
              legend=False,
              zorder=1)
for i in range(3):
    plt.errorbar(x=i,
                 y=ex_mean[i],
                 yerr=ex_se[i],
                 color='black',
                 capsize=10,
                 capthick=0.75,
                 marker='_',
                 markersize=40,
                 markeredgewidth=1,
                 zorder=2,
                 linewidth=1.5)
plt.xticks([0, 1, 2],['Missing', 'Empty', 'Border'], fontweight='bold')
plt.xlim(-0.5, 2.5)
plt.xlabel('')
plt.ylabel('Proportion of Excluded Rafts', fontweight='bold')
# sns.scatterplot(x=[0, 1, 2],
#                 y=ex_mean,
#                 c='k',
#                 marker='^',
#                 s=100,
#                 legend=False
#                )

sns.set(font_scale=1.25)
# fig.tight_layout()