#%%
import os
from pathlib import Path
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import scipy.stats as stats

#%% [markdown]
# Fluor Analysis

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

df_fluor_means = pd.DataFrame({'Array': arrs_col2,
                         'Feat': var_names_col2,
                         'Mean': means_col2,
                         'Stdev': stdevs_col2})

#%%
df_fluor_means.to_csv((dataDir / 'FluorAnalysis_Fixed_Means.csv'),
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

axes[0, 0].set_ylabel(r'Fluorescence Area ($\mathregular{10^5}$ $\mathregular{px^2}$)', fontweight='bold')
axes[0, 0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
axes[0, 0].yaxis.get_offset_text().set_visible(False)

axes[0, 1].set_ylabel('Density', fontweight='bold')

axes[1, 0].set_ylabel('Eccentricity', fontweight='bold')

axes[1, 1].set_ylabel('Radius (μm)', fontweight='bold')

plt.tight_layout()
plt.show()

#%%
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
# Raft Grading

#%%
dataDirs = [Path(r'C:\Users\MVX\OneDrive - UW\Allbritton Lab\Manuscripts\Manuscript 1 - Technical\Data\Fixed Gastruloid Fluor Analysis'),
            Path(r'C:\Users\MVX\OneDrive - UW\Allbritton Lab\Manuscripts\Manuscript 1 - Technical\Data\Eu & Aneu Gastruloid Fluor Analysis')]
gradPaths = []
for dir in dataDirs:
    gradPath = [file for file in dir.iterdir() if 'RaftGrading' in file.stem]
    gradPaths.extend(gradPath)

#%%
grads = ['Pass', 'Excl_missing', 'Excl_empty', 'Excl_border']
arrLocs = ['Center', 'Edge', 'Total']

cenRafts = []
for r in range(2, 21):
    for c in range(2, 21):
        cenRafts.append((r * 23) + (c + 1))

arrs_col = []
grads_col = []
arrLocs_col = []
counts_col = []

for idx1, gradPath in enumerate(gradPaths):
    df_grad = pd.read_csv(gradPath)

    for grad in grads:
        rafts = [raft for raft in df_grad[grad].tolist() if pd.notna(raft)]

        count = [0, 0]
        for raft in rafts:
            if raft in cenRafts:
                count[0] += 1
            else:
                count[1] += 1

        for idx2, arrLoc in enumerate(arrLocs):
            arrs_col.append(idx1 + 1)
            grads_col.append(grad)

            arrLocs_col.append(arrLoc)

            if arrLoc != 'Total':
                counts_col.append(count[idx2])
            else:
                counts_col.append(np.sum(count))

df_grad_counts = pd.DataFrame({'Array': arrs_col,
                               'Grade': grads_col,
                               'ArrayLoc': arrLocs_col,
                               'Count': counts_col})

#%%
grads_col2 = []
arrLocs_col2 = []
means_col2 = []
stdevs_col2 = []

for grad in grads:
    for arrLoc in arrLocs:
        grads_col2.append(grad)
        arrLocs_col2.append(arrLoc)

        df_sub = df_grad_counts[(df_grad_counts['Grade'] == grad) &
                                (df_grad_counts['ArrayLoc'] == arrLoc)]
        mean = np.mean(df_sub['Count'])
        stdev = np.std(df_sub['Count'])

        means_col2.append(mean)
        stdevs_col2.append(stdev)

df_grad_means = pd.DataFrame({'Grade': grads_col2,
                              'ArrayLoc': arrLocs_col2,
                              'Mean': means_col2,
                              'Stdev': stdevs_col2})

#%%
grad_labs = ['Pass', 'Missing', 'Empty', 'Border']

#%%
df_grad_counts.to_csv((dataDir.parent / 'Raft Grading Count_Raw.csv'),
                      index=False)
df_grad_means.to_csv((dataDir.parent / 'Raft Grading Count_Mean.csv'),
                     index=False)

#%%
plt.rcdefaults()
sns.set_theme(style='white', font_scale=1.25)
plt.rcParams['ytick.left'] = True
fig, axes = plt.subplots(1, 1, figsize=(8, 4))

df_sub = df_grad_counts[df_grad_counts['ArrayLoc'] == 'Total']
df_sub.reset_index(drop=True, inplace=True)
df_sub_mean = df_grad_means[df_grad_means['ArrayLoc'] == 'Total']
df_sub_mean.reset_index(drop=True, inplace=True)

sns.stripplot(ax=axes,
              x='Grade',
              y='Count',
              data=df_sub,
              hue='Array',
              size=12,
              alpha=0.75,
              jitter=0.3,
              palette='viridis',
              legend=False,
              zorder=1)
axes.errorbar(x=df_sub_mean['Grade'],
             y=df_sub_mean['Mean'],
             yerr=df_sub_mean['Stdev'],
             fmt='none',
             color='black',
             capsize=10,
             capthick=1.5,
             marker='_',
             markersize=40,
             markeredgewidth=3,
             linewidth=3,
             zorder=2)
for i, row in df_sub_mean.iterrows():
    axes.hlines(y=row['Mean'],
               xmin= (i - 0.15),
               xmax= (i + 0.15),
               colors='black',
               linewidth=3,
               zorder=3)

axes.set_xlabel('')
axes.set_xticks([0, 1, 2, 3],
                grad_labs,
                fontweight='bold')
axes.set_ylabel('Number of Rafts', fontweight='bold')
axes.set_ylim(0, (np.max(df_sub['Count']) * 1.1))

plt.tight_layout()
plt.show()

#%%
arrLocs_plot = arrLocs[:-1]

plt.rcdefaults()
sns.set_theme(style='white', font_scale=1.25)
plt.rcParams['ytick.left'] = True
fig, axes = plt.subplots(2, 2, figsize=(8, 6))

for idx, grad in enumerate(grads):
    row = int(idx / 2)
    col = idx % 2
    print(row)
    print(col)

    df_sub = df_grad_counts[(df_grad_counts['Grade'] == grad) &
                            (df_grad_counts['ArrayLoc'] != 'Total')]
    df_sub.reset_index(drop=True, inplace=True)
    df_sub_mean = df_grad_means[(df_grad_means['Grade'] == grad) &
                                (df_grad_means['ArrayLoc'] != 'Total')]
    df_sub_mean.reset_index(drop=True, inplace=True)

    sns.swarmplot(ax=axes[row, col],
                  x='ArrayLoc',
                  y='Count',
                  data=df_sub,
                  hue='Array',
                  size=10,
                  alpha=0.75,
                  palette='viridis',
                  legend=False,
                  zorder=1)

    axes[row, col].errorbar(x=df_sub_mean['ArrayLoc'],
                            y=df_sub_mean['Mean'],
                            yerr=df_sub_mean['Stdev'],
                            fmt='none',
                            color='black',
                            capsize=10,
                            capthick=2,
                            marker='_',
                            markersize=40,
                            markeredgewidth=2,
                            linewidth=3,
                            zorder=2)
    for i, row2 in df_sub_mean.iterrows():
        axes[row, col].hlines(y=row2['Mean'],
                              xmin=(i - 0.2),
                              xmax=(i + 0.2),
                              colors='black',
                              linewidth=2,
                              zorder=3)

    axes[row, col].set_xlabel('')
    axes[row, col].set_xticks([0, 1],
                              arrLocs_plot,
                              fontweight='bold')
    axes[row, col].set_xlim(-0.5, 1.5)
    axes[row, col].set_ylabel(f'{grad_labs[idx]}', fontweight='bold')
    axes[row, col].set_ylim(0, (np.max(df_sub['Count']) * 1.1))

plt.tight_layout()
plt.show()
