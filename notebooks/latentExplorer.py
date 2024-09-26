#%% [markdown]
# Notebook to analyze latent space.

#%%
import os
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import umap

#%% [markdown]
# GLR Live

#%%
dirs_df = [r'C:\Users\MVX\Documents\2 - Data\240805_R2GLR_1.8e6_0-48hrBMP4_0%aneu_4_Analysis\array1_FluorAnalysis_GLR - copy.csv',
           r'C:\Users\MVX\Documents\2 - Data\240805_R2GLR_1.8e6_0-48hrBMP4_0%aneu_4_Analysis\array2_FluorAnalysis_GLR - copy.csv',
           ]

df_lat_good = pd.DataFrame()

for arr, dir in enumerate(dirs_df):
    df_lat = pd.read_csv(dir)

    df_lat_good_dir = df_lat[(df_lat['Missing'] == 0) &
                             (df_lat['Empty'] == 0) &
                             (df_lat['Border'] == 0)]
    print(df_lat_good_dir)


    df_lat_good_dir['Array'] = [arr + 1] * df_lat_good_dir.shape[0]

    # df_lat_good_dir.head()
    # print(df_lat_good_dir.shape)

    df_lat_good = pd.concat([df_lat_good, df_lat_good_dir], axis=0)

    feats = df_lat_good.columns.tolist()

#%%
drop_list = ['Array',
             'Time',
             'Raft',
             'Missing',
             'Empty',
             'Border',
             'NumObj_Hull',
             'ThreshVal_Hull',
             'FArea_Hull',
             'Den_Hull',
             'Eccen_Hull',
             'Radius_Hull',
             'IntMean_Hull',
             'NumObj_SOX2',
             'ThreshVal_SOX2'
             ]

df_lat_drop = df_lat_good.drop(columns=drop_list)
df_lat_drop.dropna(inplace=True)

# print(df_lat_drop.shape)

#%%
df_lat_norm = StandardScaler().fit_transform(df_lat_drop)

#%%
# Create UMAP
umap_transformer = umap.UMAP(n_neighbors = 30)
umap_out = umap_transformer.fit_transform(df_lat_norm)

umap_df = pd.DataFrame(umap_out, columns=["UMAP_1","UMAP_2"])

#%%
for ft in feats:
    ft_list_good = df_lat_good[ft].tolist()
    ft_list = [ft_list_good[idx] for idx in df_lat_drop.index.tolist()]

    if 'FArea' in ft:
        umap_df[ft] = [x / 1e5 for x in ft_list]
    else:
        umap_df[ft] = ft_list

#%%
rep = ['SOX2']
feats2 = [['Time'],
          ['FArea', 'Den', 'Eccen', 'Radius'],
          ['IntMean', 'IntMean_Fold', 'IntMean_PerPx', 'IntMean_PerPx_Fold'],
          ['IntMean_RadialSlope']]

fig, axes = plt.subplots(4, 4, figsize=(18, 14))

ch = rep[0]
for i, fts in enumerate(feats2):
    for j, ft in enumerate(fts):
        if ft == 'Time':
            label = ft
        else:
            label = f'{ft}_{ch}'

        sns.scatterplot(ax=axes[i, j],
                        data=shuffle(umap_df),
                        x='UMAP_1',
                        y='UMAP_2',
                        hue=label,
                        alpha=0.5,
                        palette='viridis',
                        legend=False
                        )

        norm = plt.Normalize(umap_df[label].min(), umap_df[label].max())
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=axes[i, j])

        if 'FArea' in ft:
            unit = r'($\mathregular{10^5}$ $\mathregular{px^2}$)'
        elif 'Radius' in ft:
            unit = r'(μm)'
        elif 'RadialSlope' in ft:
            unit = r'($\mathregular{μm^{-1}}$)'
        else:
            unit = ''

        if ft == 'Time':
            axes[i, j].set_title((f'{ft} (h)'),
                                 fontweight='bold')
        else:
            axes[i, j].set_title((f'{ft} {ch} {unit}'),
                                 fontweight='bold')

for ax in axes.flat:
    if not ax.has_data():
        ax.remove()

plt.tight_layout()
plt.show()

#%% [markdown]
# Fixed

#%%
# Fixed
dirs_df = [r'C:\Users\MVX\Documents\2 - Data\Analysis\231019_RUES2_1.8e6_48hrBMP4_0%aneu_1_Analysis',
           r'C:\Users\MVX\Documents\2 - Data\Analysis\231019_RUES2_1.8e6_48hrBMP4_0%aneu_2_Analysis',
           r'C:\Users\MVX\Documents\2 - Data\Analysis\231019_RUES2_1.8e6_48hrBMP4_0%aneu_3_Analysis']

df_lat_good = pd.DataFrame()

for arr, dir in enumerate(dirs_df):
    df_lat = pd.read_csv(Path(dir) / r'array1_FluorAnalysis_Fixed.csv')

    df_lat_good_dir = df_lat[(df_lat['Missing'] == 0) &
                             (df_lat['Empty'] == 0) &
                             (df_lat['Border'] == 0) &
                             (df_lat['Lo_Int'] == 0)]
    df_lat_good_dir.dropna(inplace=True)
    df_lat_good_dir.reset_index(drop=True, inplace=True)
    df_lat_good_dir['Array'] = [arr + 1] * df_lat_good_dir.shape[0]

    # df_lat_good_dir.head()
    # print(df_lat_good_dir.shape)

    df_lat_good = pd.concat([df_lat_good, df_lat_good_dir], axis=0)

#%%
df_lat_drop = df_lat_good.drop(columns=['Array',
                                        'Time',
                                        'Raft',
                                        'Missing',
                                        'Empty',
                                        'Border',
                                        'Lo_Int',
                                        'NumObj_Blue',
                                        'ThreshVal_Blue',
                                        'NumObj_Green',
                                        'ThreshVal_Green',
                                        'NumObj_Red',
                                        'ThreshVal_Red',
                                        'NumObj_FarRed',
                                        'ThreshVal_FarRed',
                                        ])
# print(df_lat_drop.shape)

feats = df_lat_drop.columns.tolist()

#%%
df_lat_norm = StandardScaler().fit_transform(df_lat_drop)

#%%
# Create UMAP
umap_transformer = umap.UMAP(n_neighbors = 30)
umap_out = umap_transformer.fit_transform(df_lat_norm)

umap_df = pd.DataFrame(umap_out, columns=["UMAP_1","UMAP_2"])
umap_df['Rafts'] = df_lat_good['Raft'].tolist()
for ft in feats:
    ft_list = df_lat_good[ft].tolist()
    if 'FArea' in ft:
        umap_df[ft] = [x / 1e5 for x in ft_list]
    else:
        umap_df[ft] = ft_list

#%%
channels = ['Blue', 'Green', 'Red', 'FarRed']
stains = ['Hoechst', 'SOX2', 'BRA', 'CDX2']
feats = ['FArea', 'Dens', 'Eccen', 'Radius']
units = [r'($\mathregular{10^5}$ $\mathregular{px^2}$)', '', '', r'(μm)']

fig, axes = plt.subplots(5, 4, figsize=(18, 14))

label = 'FAreaHull'
sns.scatterplot(ax=axes[0, 0],
                data=shuffle(umap_df),
                x='UMAP_1',
                y='UMAP_2',
                hue=label,
                alpha=0.5,
                palette='viridis',
                legend=False
                )
norm = plt.Normalize(umap_df[label].min(), umap_df[label].max())
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=axes[0, 0])
axes[0, 0].set_title((f'FArea_Hull' + f' {units[0]}'),
                     fontweight='bold')

for i, ch in enumerate(channels):
    for j, ft in enumerate(feats):
        label = f'{ft}_{ch}'
        sns.scatterplot(ax=axes[i + 1, j],
                        data=shuffle(umap_df),
                        x='UMAP_1',
                        y='UMAP_2',
                        hue=label,
                        alpha=0.5,
                        palette='viridis',
                        legend=False
                        )

        norm = plt.Normalize(umap_df[label].min(), umap_df[label].max())
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=axes[i + 1, j])
        axes[i + 1, j].set_title((f'{ft} {channels[i]}' + f' {units[j]}'),
                             fontweight='bold')

for ax in axes.flat:
    if not ax.has_data():
        ax.remove()

plt.tight_layout()
plt.show()

#%%
label = 'FAreaHull'

plt.figure(figsize=(8, 6))
scatter = sns.scatterplot(shuffle(umap_df),
                          x="UMAP_1",
                          y="UMAP_2",
                          hue=label,
                          alpha=0.5,
                          palette="viridis",
                          legend=False)

norm = plt.Normalize(umap_df[label].min(), umap_df[label].max())
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=scatter.axes)
plt.title('Area Hull (10^5 px^2)', fontweight='bold')

plt.tight_layout()
plt.show()

#%%
label = 'Radius_Green'

plt.figure(5, 4, figsize=(8, 6))
scatter = sns.scatterplot(shuffle(umap_df),
                          x="UMAP_1",
                          y="UMAP_2",
                          hue=label,
                          alpha=0.5,
                          palette="viridis",
                          legend=False)

norm = plt.Normalize(umap_df[label].min(), umap_df[label].max())
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=scatter.axes)
plt.title('Radius SOX2 (um)', fontweight='bold')

plt.tight_layout()
plt.show()