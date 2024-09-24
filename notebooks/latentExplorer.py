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

#%%
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
    if 'FArea' in ft:
        umap_df[ft] = [x / 1e5 for x in df_lat_drop[ft].tolist()]
    else:
        umap_df[ft] = df_lat_good[ft].tolist()

#%%
channels = ['Blue', 'Green', 'Red', 'FarRed']
stains = ['Hoechst', 'SOX2', 'BRA', 'CDX2']
feats = ['FArea', 'Dens', 'Eccen', 'Radius']
units = [r'($10^5 px^2$)', '', '', r'($\mu$m)']

fig, axes = plt.subplots(4, 4, figsize=(18, 12))

for i, ch in enumerate(channels):
    for j, ft in enumerate(feats):
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
        axes[i, j].set_title((f'{ft} {channels[i]}' + f' {units[j]}'),
                             fontweight='bold')

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