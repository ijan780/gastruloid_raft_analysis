#%%
import os
from pathlib import Path
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

#%%
dataDir = Path(r'C:\Users\MVX\OneDrive - UW\Allbritton Lab\Manuscripts\Manuscript 1 - Technical\Data\Fixed Gastruloid Fluor Analysis')
os.chdir(dataDir)
arrays = [_ for _ in range(1, 4)]
pxSize = 1.3163  # um/px

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

    for ft in var_names[1]:
        df_fluor_filter[ft] = [(fArea * pxSize**2) for fArea in df_fluor_filter[ft].tolist()]

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
counts_col2 = []
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
            count = df_sub.shape[0]
            mean = np.mean(df_sub['value'])
            stdev = np.std(df_sub['value'])

            counts_col2.append(count)
            means_col2.append(mean)
            stdevs_col2.append(stdev)

df_fluor_means = pd.DataFrame({'Array': arrs_col2,
                               'Count': counts_col2,
                               'Feat': var_names_col2,
                               'Mean': means_col2,
                               'Stdev': stdevs_col2})

#%%
df_fluor_means.to_csv((dataDir.parent / 'FluorAnalysis_Fixed_Means.csv'),
                index=False)

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
    df_sub_mean = df_fluor_means[df_fluor_means['Feat'].isin(var_name)]

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

axes[0, 0].set_ylabel(r'Fluorescence Area ($\mathregular{10^5}$ $\mathregular{μm^2}$)',
                      fontweight='bold')
axes[0, 0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
axes[0, 0].yaxis.get_offset_text().set_visible(False)

axes[0, 1].set_ylabel('Density', fontweight='bold')

axes[1, 0].set_ylabel('Eccentricity', fontweight='bold')

axes[1, 1].set_ylabel('Radius (μm)', fontweight='bold')

plt.tight_layout()
plt.show()

#%%
#Need to comment out lo_int filter

plt.rcdefaults()
sns.set_theme(style='white', font_scale=1.25)
plt.rcParams['ytick.left'] = True
fig, axes = plt.subplots(1, 1, figsize=(6, 4))

df_sub = df_fluors[df_fluors['variable'].isin(var_names[0])]
df_sub_mean = df_fluor_means[df_fluor_means['Feat'].isin(var_names[0])]

sns.stripplot(ax=axes,
              x='variable',
              y='value',
              data=df_sub,
              hue='Array',
              size=3,
              alpha=0.5,
              palette='colorblind',
              jitter=0.3,
              legend=False)
sns.swarmplot(ax=axes,
              x='Feat',
              y='Mean',
              data=df_sub_mean,
              hue='Array',
              size=15,
              palette='colorblind',
              edgecolor='k',
              linewidth=2,
              legend=False)
axes.set_xlabel('')
axes.set_xticks([0, 1, 2, 3], ['Hoechst', 'SOX2', 'BRA', 'CDX2'], fontweight='bold')
axes.set_ylabel('Threshold', fontweight='bold')

plt.tight_layout()
plt.show()

#%% [markdown]
# 12wp vs raft comparison

#%%
from pathlib import Path
import pandas as pd
import seaborn as sns

#%%
dataDir = Path(r'C:\Users\MVX\OneDrive - UW\Allbritton Lab\Confocal\241210_RUES2_12wp vs Raft')
df_grad = pd.read_csv(dataDir / '12wp vs Raft Gastruloid Grading Count.csv')
grads = ['Pass', 'Malformed']
arrLocs = ['Center', 'Edge']

#%%
df_wp = df_grad[df_grad['Surface'] == '12wp']

plt.rcdefaults()
sns.set_theme(style='white', font_scale=1.25)
plt.rcParams['ytick.left'] = True
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

for idx, grad in enumerate(grads):
    df_sub = df_wp[df_wp['Grade'] == grad]

    sns.swarmplot(ax=axes[idx],
                  x='ArrayLoc',
                  y='PercRaft',
                  data=df_sub,
                  hue='Replicate',
                  size=15,
                  alpha=0.75,
                  palette='colorblind',
                  legend=False,
                  zorder=1)

    axes[idx].set_xlabel('')
    axes[idx].set_xticks([0, 1],
                         arrLocs,
                         fontweight='bold')
    axes[idx].set_xlim(-0.5, 1.5)
    axes[idx].set_ylabel(f'% {grad}', fontweight='bold')
    axes[idx].set_ylim(0, 115)

    if grad == 'Pass':
        axes[idx].plot([0, 0, 1, 1],
                       [106,
                        108,
                        108,
                        106],
                       linewidth=2,
                       color='k')

        axes[idx].text((0 + 1) * 0.5,
                       108,
                       'n.s.',
                       ha='center',
                       va='bottom',
                       color='k',
                       fontsize=16
                       )
    elif grad == 'Malformed':
        axes[idx].plot([0, 0, 1, 1],
                       [6,
                        8,
                        8,
                        6],
                       linewidth=2,
                       color='k')

        axes[idx].text((0 + 1) * 0.5,
                       8,
                       'n.s.',
                       ha='center',
                       va='bottom',
                       color='k',
                       fontsize=16
                       )

plt.tight_layout()
plt.show()

#%%
df_wp = df_grad[df_grad['Surface'] == 'Raft']

plt.rcdefaults()
sns.set_theme(style='white', font_scale=1.25)
plt.rcParams['ytick.left'] = True
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

for idx, grad in enumerate(grads):
    df_sub = df_wp[df_wp['Grade'] == grad]

    sns.swarmplot(ax=axes[idx],
                  x='ArrayLoc',
                  y='PercRaft',
                  data=df_sub,
                  marker='s',
                  hue='Replicate',
                  size=15,
                  alpha=0.75,
                  palette='colorblind',
                  legend=False,
                  zorder=1)

    axes[idx].set_xlabel('')
    axes[idx].set_xticks([0, 1],
                         arrLocs,
                         fontweight='bold')
    axes[idx].set_xlim(-0.5, 1.5)
    axes[idx].set_ylabel(f'% {grad}', fontweight='bold')
    axes[idx].set_ylim(0, 115)

    if grad == 'Pass':
        axes[idx].plot([0, 0, 1, 1],
                       [106,
                        108,
                        108,
                        106],
                       linewidth=2,
                       color='k')

        axes[idx].text((0 + 1) * 0.5,
                       108,
                       'n.s.',
                       ha='center',
                       va='bottom',
                       color='k',
                       fontsize=16
                       )
    elif grad == 'Malformed':
        axes[idx].plot([0, 0, 1, 1],
                       [6,
                        8,
                        8,
                        6],
                       linewidth=2,
                       color='k')

        axes[idx].text((0 + 1) * 0.5,
                       8,
                       'n.s.',
                       ha='center',
                       va='bottom',
                       color='k',
                       fontsize=16
                       )

plt.tight_layout()
plt.show()