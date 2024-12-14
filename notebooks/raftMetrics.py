#%% [markdown]
# Bead distribution analysis

#%%
from pathlib import Path
import tifffile as tiff
import numpy as np
from skimage.util import invert
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import binary_erosion, remove_small_holes, remove_small_objects
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt
import pandas as pd
import matplotlib.pyplot as plt

#%%
dataDir = Path(r'C:\Users\MVX\OneDrive - UW\Allbritton Lab\DIC\3. 800x800x104um\241029_Beads_Compiled')
arrs = [1, 2, 3]
locs = ['center', 'edge']
px_size = 1.59170008 # um/px

arrs_col = []
locs_col = []
rafts_col = []
raftAreas_col = []
raftCens_col = []
beadAreas_col = []
beadDens_col = []
fromEdges_mean_col = []
fromEdges_stdev_col = []

imgPaths = [file for file in dataDir.iterdir() if file.suffix == '.tif']

for arr in arrs:
    for loc in locs:
        imgPaths_bf = [file for file in imgPaths if (f'arr{arr}' in file.stem
                                                     and loc in file.stem
                                                     and 'bf' in file.stem)]
        imgPaths_tx = [file for file in imgPaths if (f'arr{arr}' in file.stem
                                                     and loc in file.stem
                                                     and 'tx' in file.stem)]
        # print(len(imgPaths_bf) == len(imgPaths_tx))
        raft_count = 0

        for i in range(len(imgPaths_bf)):
            path_bf = imgPaths_bf[i]
            path_tx = imgPaths_tx[i]

            # Segment rafts from BF images to create individual masks.
            with tiff.TiffFile(path_bf) as tif:
                img_bf = tif.asarray()
            img_bf = gaussian(img_bf, sigma=2, preserve_range=True)
            inv_img_bf = invert(img_bf)
            mask_bf = inv_img_bf > threshold_otsu(inv_img_bf)
            mask_bf = binary_erosion(mask_bf, footprint=np.ones((9, 9)))
            mask_bf = remove_small_holes(mask_bf, area_threshold=2.5e5)
            mask_bf = remove_small_objects(mask_bf, min_size=2.3e5)
            # plt.imshow(mask_bf, cmap='gray', interpolation='nearest')
            # plt.axis('off')
            # plt.show()

            labels = label(mask_bf)
            unique_labels = np.unique(labels)[1:] # background label is 0
            regions = regionprops(labels)

            masks = {}
            for lab in unique_labels:
                masks[lab] = np.where(labels == lab, 1, 0).astype(int)

            # fig, axes = plt.subplots(1, len(unique_labels) + 1, figsize=(12, 4))
            # axes[0].imshow(labels, cmap='nipy_spectral')
            # axes[0].set_title('Labeled Image')
            # axes[0].axis('off')
            #
            # for j, lab in enumerate(unique_labels, start=1):
            #     axes[j].imshow(masks[lab], cmap='gray')
            #     axes[j].set_title(f'Mask for Object {lab}')
            #     axes[j].axis('off')
            #
            # plt.show()

            # Apply raft masks to TX image.
            with tiff.TiffFile(path_tx) as tif:
                img_tx = tif.asarray()
            img_tx = gaussian(img_tx, sigma=2, preserve_range=True)
            mask_tx = img_tx > 360

            # Measure bead density & distribution from raft centroid for all rafts.
            for j, lab in enumerate(unique_labels, start=1):
                raft_count += 1
                arrs_col.append(arr)
                locs_col.append(loc)
                rafts_col.append(raft_count)

                # Raft area & centroid.
                raft_region = regions[j - 1]
                raftArea = raft_region.area
                raftAreas_col.append(raftArea)
                raft_y, raft_x = raft_region.centroid
                raftCens_col.append([raft_y, raft_x])

                # Apply raft mask to bead mask.
                mask_raft = masks[lab]
                # plt.imshow(mask_raft, cmap='gray', interpolation='nearest')
                # plt.axis('off')
                # plt.show()

                img_tx_masked = np.logical_and(mask_raft, mask_tx)
                # plt.imshow(img_tx_masked, cmap='gray', interpolation='nearest')
                # plt.axis('off')
                # plt.show()

                # Bead area & density.
                beadArea = np.sum(img_tx_masked)
                beadAreas_col.append(beadArea)
                beadDens = beadArea / raftArea
                beadDens_col.append(beadDens)

                # Bead distribution.

                # Distance map for raft edges.
                mask_raft_pad = np.pad(mask_raft, pad_width=5, mode='constant', constant_values=0)
                distance_to_edge = distance_transform_edt(mask_raft_pad) # distance-to-edge map for raft mask

                # plt.figure(figsize=(6, 6))
                # ax = plt.gca()
                # im = ax.imshow(distance_to_edge, cmap='viridis', interpolation='nearest')
                # plt.axis('off')
                # divider = make_axes_locatable(ax)
                # cax = divider.append_axes("right", size="5%", pad=0.1)
                # plt.colorbar(im, cax=cax, label='Distance to Nearest Edge (px)')
                # plt.tight_layout()
                # plt.show()

                img_tx_masked_pad = np.pad(img_tx_masked, pad_width=5, mode='constant', constant_values=0)
                non_zero_coords = np.argwhere(img_tx_masked_pad > 0)  # coord of all non-zero px
                distances = [distance_to_edge[y,x] for y, x in non_zero_coords]

                fromEdge_mean = np.mean(distances) * px_size
                fromEdge_stdev = np.std(distances) * px_size
                fromEdges_mean_col.append(fromEdge_mean)
                fromEdges_stdev_col.append(fromEdge_stdev)

df_beads = pd.DataFrame({'Array': arrs_col,
                         'Location': locs_col,
                         'Raft': rafts_col,
                         'RaftArea': raftAreas_col,
                         'RaftCen': raftCens_col,
                         'BeadArea': beadAreas_col,
                         'BeadDens': beadDens_col,
                         'FromEdge_Mean': fromEdges_mean_col,
                         'FromEdge_Stdev': fromEdges_stdev_col
                         })

df_beads.to_csv((dataDir / 'Bead Distribution.csv'),
                index=False)

#%%
arrs_col2 = []
locs_col2 = []
rafts_col2 = []
beadDens_mean_col2 = []
beadDens_stdev_col2 = []
fromEdges_mean2_col2 = []
fromEdges_stdev2_col2 = []

for arr in arrs:
    for loc in locs:
        df_sub = df_beads[(df_beads['Array'] == arr) &
                          (df_beads['Location'] == loc)
                          ]
        beadDens_mean = np.mean(df_sub['BeadDens'].tolist())
        beadDens_stdev = np.std(df_sub['BeadDens'].tolist())
        fromEdge_mean2 = np.mean(df_sub['FromEdge_Mean'].tolist())
        fromEdge_stdev2 = np.std(df_sub['FromEdge_Mean'].tolist())

        arrs_col2.append(arr)
        locs_col2.append(loc)
        beadDens_mean_col2.append(beadDens_mean)
        beadDens_stdev_col2.append(beadDens_stdev)
        fromEdges_mean2_col2.append(fromEdge_mean2)
        fromEdges_stdev2_col2.append(fromEdge_stdev2)

df_beads_mean = pd.DataFrame({'Array': arrs_col2,
                              'Location': locs_col2,
                              'BeadDens_Mean': beadDens_mean_col2,
                              'BeadDens_Stdev': beadDens_stdev_col2,
                              'FromEdge_Mean': fromEdges_mean2_col2,
                              'FromEdge_Stdev': fromEdges_stdev2_col2
                              })

df_beads_mean.to_csv((dataDir / 'Bead Distribution_Means.csv'),
                index=False)

#%%
dataDir = Path(r'C:\Users\MVX\OneDrive - UW\Allbritton Lab\DIC\3. 800x800x104um\241029_Beads_Compiled')
df_beads = pd.read_csv(dataDir / 'Bead Distribution.csv')
df_beads_mean = pd.read_csv(dataDir / 'Bead Distribution_Means.csv')

#%%
import scipy.stats as stats

df_center = df_beads_mean[(df_beads_mean['Location'] == 'center')]
df_edge = df_beads_mean[(df_beads_mean['Location'] == 'edge')]

ttest_beadDens = stats.ttest_ind(a=df_center['BeadDens_Mean'],
                                 b=df_edge['BeadDens_Mean'],
                                 equal_var=True)
print(f'Bead Area Dens: {ttest_beadDens}')

ttest_fromEdge = stats.ttest_ind(a=df_center['FromEdge_Mean'],
                                 b=df_edge['FromEdge_Mean'],
                                 equal_var=True)
print(f'From Edge: {ttest_fromEdge}')

#%%
import seaborn as sns

plt.rcdefaults()
sns.set_theme(style='white', font_scale=1.25)
plt.rcParams['ytick.left'] = True
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

sns.stripplot(ax=axes[0],
              x='Location',
              y='BeadDens',
              data = df_beads,
              hue='Array',
              size=10,
              palette='colorblind',
              alpha=0.5,
              jitter=0.2,
              legend=False,
              zorder=1)
sns.swarmplot(ax=axes[0],
              x='Location',
              y='BeadDens_Mean',
              data=df_beads_mean,
              hue='Array',
              size=15,
              palette='colorblind',
              edgecolor='k',
              linewidth=2,
              legend=False,
              zorder=2)
axes[0].set_xticks([0, 1], ['Center', 'Edge'], fontweight='bold')
axes[0].set_xlabel('', fontweight='bold')
axes[0].set_xlim(-0.5,1.5)
axes[0].set_ylabel('Bead Density', fontweight='bold')
minMax = [min(df_beads['BeadDens']), max(df_beads['BeadDens'])]
axes[0].set_ylim(minMax[0] * 0.9,
                 minMax[1] * 1.12)
axes[0].yaxis.set_major_locator(plt.MaxNLocator(nbins=4))

axes[0].plot([0, 0, 1, 1],
             [minMax[1] * 1.05,
              minMax[1] * 1.07,
              minMax[1] * 1.07,
              minMax[1] * 1.05],
             linewidth=2,
             color='k')
axes[0].text((0 + 1) * 0.5,
             minMax[1] * 1.07,
             "n.s.",
             ha='center',
             va='bottom',
             color='k',
             fontsize=16)

sns.stripplot(ax=axes[1],
              x='Location',
              y='FromEdge_Mean',
              data = df_beads,
              hue='Array',
              size=10,
              palette='colorblind',
              alpha=0.5,
              jitter=0.2,
              legend=False,
              zorder=1)
sns.swarmplot(ax=axes[1],
              x='Location',
              y='FromEdge_Mean',
              data=df_beads_mean,
              hue='Array',
              size=15,
              palette='colorblind',
              edgecolor='k',
              linewidth=2,
              legend=False,
              zorder=2)
axes[1].set_xticks([0, 1], ['Center', 'Edge'], fontweight='bold')
axes[1].set_xlabel('', fontweight='bold')
axes[1].set_xlim(-0.5, 1.5)
axes[1].set_ylabel('Distance from Edge (μm)', fontweight='bold')
minMax = [min(df_beads['FromEdge_Mean']), max(df_beads['FromEdge_Mean'])]
axes[1].set_ylim(minMax[0] * 0.9,
                 minMax[1] * 1.12)
axes[1].yaxis.set_major_locator(plt.MaxNLocator(nbins=4))

axes[1].plot([0, 0, 1, 1],
             [minMax[1] * 1.05,
              minMax[1] * 1.07,
              minMax[1] * 1.07,
              minMax[1] * 1.05],
             linewidth=2,
             color='k')
axes[1].text((0 + 1) * 0.5,
             minMax[1] * 1.07,
             "n.s.",
             ha='center',
             va='bottom',
             color='k',
             fontsize=16)

plt.tight_layout()
plt.show()

#%% [markdown]
# Raft dimensions analysis.

#%%
from pathlib import Path
import pandas as pd
import numpy as np

dataDir = Path(r'C:\Users\MVX\OneDrive - UW\Allbritton Lab\Confocal\241106_800x800x104um Raft')
df_edges = pd.read_csv(dataDir / r'800x800x104um Raft Measurements_Raw_Edge.csv')
df_thicks = pd.read_csv(dataDir / r'800x800x104um Raft Measurements_Raw_Thickness.csv')

arrays = [_ for _ in range(1,4)]
arr_locs = ['Center', 'Edge']
raft_locs = ['Center', 'Edge']
rafts = [_ for _ in range(1,6)]

arrs_edge_col = []
arr_locs_edge_col = []
rafts_edge_col = []
measures_edge_col = []
stdev_edge_col = []

for arr in arrays:
    for arr_loc in arr_locs:
        for raft in rafts:
            arrs_edge_col.append(arr)
            arr_locs_edge_col.append(arr_loc)
            rafts_edge_col.append(raft)

            df_sub = df_edges[(df_edges['Array'] == arr) &
                              (df_edges['ArrayLoc'] == arr_loc) &
                              (df_edges['Raft'] == raft)]
            measure_edge = np.mean(df_sub['Measurement'].tolist())
            measures_edge_col.append(measure_edge)
            stdev_edge = np.std(df_sub['Measurement'].tolist())
            stdev_edge_col.append(stdev_edge)

df_edges_means = pd.DataFrame({'Array': arrs_edge_col,
                               'ArrayLoc': arr_locs_edge_col,
                               'Raft': rafts_edge_col,
                               'Measurement': measures_edge_col,
                               'Stdev': stdev_edge_col})

arrs_thick_col = []
arr_locs_thick_col = []
rafts_thick_col = []
raft_locs_thick_col = []
measures_thick_col = []
stdev_thick_col = []
for arr in arrays:
    for arr_loc in arr_locs:
        for raft in rafts:
            for raft_loc in raft_locs:
                arrs_thick_col.append(arr)
                arr_locs_thick_col.append(arr_loc)
                rafts_thick_col.append(raft)
                raft_locs_thick_col.append(raft_loc)

                df_sub = df_thicks[(df_thicks['Array'] == arr) &
                                   (df_thicks['ArrayLoc'] == arr_loc) &
                                   (df_thicks['Raft'] == raft) &
                                   (df_thicks['RaftLoc'] == raft_loc)]
                measure_thick = np.mean(df_sub['Measurement'].tolist())
                measures_thick_col.append(measure_thick)
                stdev_thick = np.std(df_sub['Measurement'].tolist())
                stdev_thick_col.append(stdev_thick)
df_thicks_means = pd.DataFrame({'Array': arrs_thick_col,
                                'ArrayLoc': arr_locs_thick_col,
                                'Raft': rafts_thick_col,
                                'RaftLoc': raft_locs_thick_col,
                                'Measurement': measures_thick_col,
                                'Stdev': stdev_thick_col})

#%%
df_edges_means.to_csv((dataDir / '800x800x104um Raft Measurements_Mean_Edge.csv'),
                      index=False)
df_thicks_means.to_csv((dataDir / '800x800x104um Raft Measurements_Mean_Thickness.csv'),
                       index=False)

#%%
arrs_edge_col2 = []
arr_locs_edge_col2 = []
means_edge_col2 = []
stdev_edge_col2 = []

for arr in arrays:
    for arr_loc in (arr_locs + ['Total']):
        arrs_edge_col2.append(arr)
        arr_locs_edge_col2.append(arr_loc)

        if arr_loc != 'Total':
            df_sub = df_edges_means[(df_edges_means['Array'] == arr) &
                                    (df_edges_means['ArrayLoc'] == arr_loc)]
        else:
            df_sub = df_edges_means[df_edges_means['Array'] == arr]
        mean = np.mean(df_sub['Measurement'].tolist())
        stdev = np.std(df_sub['Measurement'].tolist())
        means_edge_col2.append(mean)
        stdev_edge_col2.append(stdev)

df_edges_means2 = pd.DataFrame({'Array': arrs_edge_col2,
                                'ArrayLoc': arr_locs_edge_col2,
                                'Mean': means_edge_col2,
                                'Stdev': stdev_edge_col2})

arrs_thick_col2 = []
arr_locs_thick_col2 = []
raft_locs_thick_col2 = []
means_thick_col2 = []
stdev_thick_col2 = []

for arr in arrays:
    for arr_loc in (arr_locs + ['Total']):
        for raft_loc in raft_locs:
            arrs_thick_col2.append(arr)
            arr_locs_thick_col2.append(arr_loc)
            raft_locs_thick_col2.append(raft_loc)

            if arr_loc != 'Total':
                df_sub = df_thicks_means[(df_thicks_means['Array'] == arr) &
                                         (df_thicks_means['ArrayLoc'] == arr_loc) &
                                         (df_thicks_means['RaftLoc'] == raft_loc)]
            else:
                df_sub = df_thicks_means[(df_thicks_means['Array'] == arr) &
                                         (df_thicks_means['RaftLoc'] == raft_loc)]
            mean = np.mean(df_sub['Measurement'].tolist())
            stdev = np.std(df_sub['Measurement'].tolist())
            means_thick_col2.append(mean)
            stdev_thick_col2.append(stdev)

df_thicks_means2 = pd.DataFrame({'Array': arrs_thick_col2,
                                 'ArrayLoc': arr_locs_thick_col2,
                                 'RaftLoc': raft_locs_thick_col2,
                                 'Mean': means_thick_col2,
                                 'Stdev': stdev_thick_col2})

#%%
df_edges_means2.to_csv((dataDir / '800x800x104um Raft Measurements_Mean_Edge_Means.csv'),
                      index=False)
df_thicks_means2.to_csv((dataDir / '800x800x104um Raft Measurements_Mean_Thickness_Means.csv'),
                       index=False)

#%%
import scipy.stats as stats

arrCenters_edge = df_edges_means2[df_edges_means2['ArrayLoc'] == 'Center']['Mean']
arrEdges_edge = df_edges_means2[df_edges_means2['ArrayLoc'] == 'Edge']['Mean']
ttest_edge = stats.ttest_ind(a=arrCenters_edge, b=arrEdges_edge, equal_var=True)
print(f'Raft Edges: {ttest_edge}')

center_arrCenters_thick = df_thicks_means2[(df_thicks_means2['ArrayLoc'] == 'Center') &
                                           (df_thicks_means2['RaftLoc'] == 'Center')]['Mean']
center_arrEdges_thick = df_thicks_means2[(df_thicks_means2['ArrayLoc'] == 'Edge') &
                                         (df_thicks_means2['RaftLoc'] == 'Center')]['Mean']
ttest_center_thick = stats.ttest_ind(a=center_arrCenters_thick, b=center_arrEdges_thick, equal_var=True)
print(f'Raft Center Thickness: {ttest_center_thick}')

edge_arrCenters_thick = df_thicks_means2[(df_thicks_means2['ArrayLoc'] == 'Center') &
                                         (df_thicks_means2['RaftLoc'] == 'Edge')]['Mean']
edge_arrEdges_thick = df_thicks_means2[(df_thicks_means2['ArrayLoc'] == 'Edge') &
                                       (df_thicks_means2['RaftLoc'] == 'Edge')]['Mean']
ttest_edge_thick = stats.ttest_ind(a=edge_arrCenters_thick, b=edge_arrEdges_thick, equal_var=True)
print(f'Raft Edge Thickness: {ttest_edge_thick}')

#%%
arr_locs_edge_col3 = []
means_edge_col3 = []
stdev_edge_col3 = []

for arr_loc in (arr_locs + ['Total']):
    arr_locs_edge_col3.append(arr_loc)

    df_sub = df_edges_means2[df_edges_means2['ArrayLoc'] == arr_loc]

    mean = np.mean(df_sub['Mean'].tolist())
    stdev = np.std(df_sub['Mean'].tolist())
    means_edge_col3.append(mean)
    stdev_edge_col3.append(stdev)

df_edges_means3 = pd.DataFrame({'ArrayLoc': arr_locs_edge_col3,
                                'Mean': means_edge_col3,
                                'Stdev': stdev_edge_col3})

arr_locs_thick_col3 = []
raft_locs_thick_col3 = []
means_thick_col3 = []
stdev_thick_col3 = []

for arr_loc in (arr_locs + ['Total']):
    for raft_loc in raft_locs:
        arr_locs_thick_col3.append(arr_loc)
        raft_locs_thick_col3.append(raft_loc)

        df_sub = df_thicks_means2[(df_thicks_means2['ArrayLoc'] == arr_loc) &
                                  (df_thicks_means2['RaftLoc'] == raft_loc)]

        mean = np.mean(df_sub['Mean'].tolist())
        stdev = np.std(df_sub['Mean'].tolist())
        means_thick_col3.append(mean)
        stdev_thick_col3.append(stdev)

df_thicks_means3 = pd.DataFrame({'ArrayLoc': arr_locs_thick_col3,
                                 'RaftLoc': raft_locs_thick_col3,
                                 'Mean': means_thick_col3,
                                 'Stdev': stdev_thick_col3})

#%%
df_edges_means3.to_csv((dataDir / '800x800x104um Raft Measurements_Mean_Edge_Means2.csv'),
                      index=False)
df_thicks_means3.to_csv((dataDir / '800x800x104um Raft Measurements_Mean_Thickness_Means2.csv'),
                       index=False)

#%%
import seaborn as sns
from matplotlib import pyplot as plt

plt.rcdefaults()
sns.set_theme(style='white', font_scale=1.25)
plt.rcParams['ytick.left'] = True
fig, axes = plt.subplots(1, 1, figsize=(4, 4))

sns.stripplot(ax=axes,
              x='ArrayLoc',
              y='Measurement',
              data=df_edges_means,
              hue='Array',
              size=10,
              palette='colorblind',
              alpha=0.5,
              jitter=0.2,
              legend=False,
              zorder=1)
sns.swarmplot(ax=axes,
              x='ArrayLoc',
              y='Mean',
              data=df_edges_means2,
              hue='Array',
              size=15,
              palette='colorblind',
              edgecolor='k',
              linewidth=2,
              legend=False,
              zorder=2)
axes.set_xticks([0, 1], arr_locs, fontweight='bold')
axes.set_xlabel('', fontweight='bold')
axes.set_xlim(-0.5, 1.5)
axes.set_ylabel('Raft Edge Length (μm)', fontweight='bold')
minMax = [min(df_edges_means['Measurement']), max(df_edges_means['Measurement'])]
axes.set_ylim(min(df_edges_means['Measurement']) * 0.998,
              max(df_edges_means['Measurement']) * 1.004)
axes.yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=4))

axes.plot([0, 0, 1, 1],
          [minMax[1] + 1,
           minMax[1] + 1.5,
           minMax[1] + 1.5,
           minMax[1] + 1],
          linewidth=2,
          color='k')
axes.text((0 + 1) * 0.5,
          minMax[1] + 1.5,
          "n.s.",
          ha='center',
          va='bottom',
          color='k',
          fontsize=16)

plt.tight_layout()
plt.show()

#%%
plt.rcdefaults()
sns.set_theme(style='white', font_scale=1.25)
plt.rcParams['ytick.left'] = True
fig, axes = plt.subplots(1, 2, figsize=(6, 4))

df_raftCen = df_thicks_means[df_thicks_means['RaftLoc'] == 'Center']
df_raftCen_mean = df_thicks_means2[df_thicks_means2['RaftLoc'] == 'Center']
sns.stripplot(ax=axes[0],
              x='ArrayLoc',
              y='Measurement',
              data = df_raftCen,
              hue='Array',
              size=10,
              palette='colorblind',
              alpha=0.5,
              jitter=0.2,
              legend=False,
              zorder=1)
sns.swarmplot(ax=axes[0],
              x='ArrayLoc',
              y='Mean',
              data=df_raftCen_mean,
              hue='Array',
              size=15,
              palette='colorblind',
              edgecolor='k',
              linewidth=2,
              legend=False,
              zorder=2)
axes[0].set_xticks([0, 1], arr_locs, fontweight='bold')
axes[0].set_xlabel('', fontweight='bold')
axes[0].set_xlim(-0.5,1.5)
minMax = [min(df_raftCen['Measurement']), max(df_raftCen['Measurement'])]
axes[0].set_ylabel('Raft Center Thickness (μm)', fontweight='bold')
axes[0].set_ylim(minMax[0] * 0.96,
                 minMax[1] * 1.065)
axes[0].yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=4))

axes[0].plot([0, 0, 1, 1],
             [minMax[1] + 1,
              minMax[1] + 1.5,
              minMax[1] + 1.5,
              minMax[1] + 1],
             linewidth=2,
             color='k')
axes[0].text((0 + 1) * 0.5,
             minMax[1] + 1.5,
             "n.s.",
             ha='center',
             va='bottom',
             color='k',
             fontsize=16)

df_raftEdge = df_thicks_means[df_thicks_means['RaftLoc'] == 'Edge']
df_raftEdge_mean = df_thicks_means2[df_thicks_means2['RaftLoc'] == 'Edge']
sns.stripplot(ax=axes[1],
              x='ArrayLoc',
              y='Measurement',
              data = df_raftEdge,
              hue='Array',
              size=10,
              palette='colorblind',
              alpha=0.5,
              jitter=0.2,
              legend=False,
              zorder=1)
sns.swarmplot(ax=axes[1],
              x='ArrayLoc',
              y='Mean',
              data=df_raftEdge_mean,
              hue='Array',
              size=15,
              palette='colorblind',
              edgecolor='k',
              linewidth=2,
              legend=False,
              zorder=2)
axes[1].set_xticks([0, 1], arr_locs, fontweight='bold')
axes[1].set_xlabel('', fontweight='bold')
axes[1].set_xlim(-0.5,1.5)
axes[1].set_ylabel('Raft Edge Thickness (μm)', fontweight='bold')
minMax = [min(df_raftEdge['Measurement']), max(df_raftEdge['Measurement'])]
axes[1].set_ylim(minMax[0] * 0.99,
                 minMax[1] * 1.025)
axes[1].yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=4))

axes[1].plot([0, 0, 1, 1],
             [minMax[1] + 1,
              minMax[1] + 1.5,
              minMax[1] + 1.5,
              minMax[1] + 1],
             linewidth=2,
             color='k')
axes[1].text((0 + 1) * 0.5,
             minMax[1] + 1.5,
             "n.s.",
             ha='center',
             va='bottom',
             color='k',
             fontsize=16
             )

plt.tight_layout()
plt.show()

#%% [markdown]
# Raft grading

#%%
from pathlib import Path
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

# dataDirs = [Path(r'C:\Users\MVX\OneDrive - UW\Allbritton Lab\Manuscripts\Manuscript 1 - Technical\Data\Eu & Aneu Gastruloid Fluor Analysis')]
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
arrTypes_col = []
grads_col = []
arrLocs_col = []
counts_col = []

for idx1, gradPath in enumerate(gradPaths):
    df_grad = pd.read_csv(gradPath)

    for grad in grads:
        if grad == 'Pass':
            # Include lo_int rafts for only fixed arrays (Hoechst still usable)
            if idx1 < 3:
                rafts = ([int(raft) for raft in df_grad['Pass'].tolist() if pd.notna(raft)] +
                         [int(raft) for raft in df_grad['Fail'].tolist() if pd.notna(raft)] +
                         [int(raft) for raft in df_grad['Excl_lo_int'].tolist() if pd.notna(raft)])
            else:
                rafts = ([int(raft) for raft in df_grad['Pass'].tolist() if pd.notna(raft)] +
                         [int(raft) for raft in df_grad['Fail'].tolist() if pd.notna(raft)])
            print((idx1 + 1), len(rafts))
        else:
            rafts = [int(raft) for raft in df_grad[grad].tolist() if pd.notna(raft)]

        count = [0, 0]
        for raft in rafts:
            if raft in cenRafts:
                count[0] += 1
            else:
                count[1] += 1

        for idx2, arrLoc in enumerate(arrLocs):
            if idx1 < 3:
                arrTypes_col.append('Fixed')
            else:
                arrTypes_col.append('Live')

            arrs_col.append(idx1 + 1)
            grads_col.append(grad)

            arrLocs_col.append(arrLoc)

            if arrLoc != 'Total':
                counts_col.append(count[idx2])
            else:
                counts_col.append(np.sum(count))

# %%
percRafts_col = []
for idx, arrLoc in enumerate(arrLocs_col):
    percRaft = 0
    if arrLoc == 'Total':
        percRaft = counts_col[idx] / 23**2
    elif arrLoc == 'Center':
        percRaft = counts_col[idx] / len(cenRafts)
    elif arrLoc == 'Edge':
        percRaft = counts_col[idx] / (23**2 - len(cenRafts))
    percRafts_col.append(percRaft * 100)

df_grad_counts = pd.DataFrame({'Array': arrs_col,
                               'ArrType': arrTypes_col,
                               'Grade': grads_col,
                               'ArrayLoc': arrLocs_col,
                               'Count': counts_col,
                               'PercRaft': percRafts_col})

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
        mean = np.mean(df_sub['PercRaft'])
        stdev = np.std(df_sub['PercRaft'])

        means_col2.append(mean)
        stdevs_col2.append(stdev)

df_grad_means = pd.DataFrame({'Grade': grads_col2,
                              'ArrayLoc': arrLocs_col2,
                              'Mean': means_col2,
                              'Stdev': stdevs_col2})

#%%
grad_labs = ['Pass', 'Missing', 'Empty', 'Border']

#%%
df_grad_counts.to_csv((dataDirs[0].parent / 'Raft Grading Count_Raw.csv'),
                      index=False)
df_grad_means.to_csv((dataDirs[0].parent / 'Raft Grading Count_Mean.csv'),
                     index=False)

#%%
plt.rcdefaults()
sns.set_theme(style='white', font_scale=1.25)
plt.rcParams['ytick.left'] = True
fig, axes = plt.subplots(1, 1, figsize=(5, 4))

df_sub = df_grad_counts[df_grad_counts['ArrayLoc'] == 'Total']
df_sub.reset_index(drop=True, inplace=True)
df_sub_mean = df_grad_means[df_grad_means['ArrayLoc'] == 'Total']
df_sub_mean.reset_index(drop=True, inplace=True)

sns.stripplot(ax=axes,
              x='Grade',
              y='PercRaft',
              data=df_sub[df_sub['ArrType'] == 'Fixed'],
              hue='Array',
              size=10,
              alpha=0.75,
              marker='o',
              jitter=0.3,
              palette='viridis',
              legend=False,
              zorder=1)
sns.stripplot(ax=axes,
              x='Grade',
              y='PercRaft',
              data=df_sub[df_sub['ArrType'] == 'Live'],
              hue='Array',
              size=10,
              alpha=0.75,
              marker='^',
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
             capthick=3,
             linewidth=3,
             zorder=2)
for i, row in df_sub_mean.iterrows():
    axes.hlines(y=row['Mean'],
               xmin= (i - 0.4),
               xmax= (i + 0.4),
               colors='black',
               linewidth=3,
               zorder=3)

axes.set_xlabel('')
axes.set_xticks([0, 1, 2, 3],
                grad_labs,
                fontweight='bold')
axes.set_ylabel('% Rafts', fontweight='bold')
axes.set_ylim(0, (np.max(df_sub['PercRaft']) * 1.1))

plt.tight_layout()
plt.show()

#%%
import scipy.stats as stats

for grad in grads:
    df_sub = df_grad_counts[df_grad_counts['Grade'] == grad]
    df_center = df_sub[df_sub['ArrayLoc'] == 'Center']
    df_edge = df_sub[df_sub['ArrayLoc'] == 'Edge']
    ttest = stats.ttest_ind(a=df_center['PercRaft'],
                            b=df_edge['PercRaft'],
                            equal_var=True)
    print(f'{grad}: {ttest}')

#%%
arrLocs_plot = arrLocs[:-1]
sig = ['n.s.', 'n.s.', 'n.s.', 'n.s.']

plt.rcdefaults()
sns.set_theme(style='white', font_scale=1.25)
plt.rcParams['ytick.left'] = True
fig, axes = plt.subplots(2, 2, figsize=(8, 6))

for idx, grad in enumerate(grads):
    row = int(idx / 2)
    col = idx % 2

    df_sub = df_grad_counts[(df_grad_counts['Grade'] == grad) &
                            (df_grad_counts['ArrayLoc'] != 'Total')]
    df_sub.reset_index(drop=True, inplace=True)
    df_sub_mean = df_grad_means[(df_grad_means['Grade'] == grad) &
                                (df_grad_means['ArrayLoc'] != 'Total')]
    df_sub_mean.reset_index(drop=True, inplace=True)

    sns.stripplot(ax=axes[row, col],
                  x='ArrayLoc',
                  y='PercRaft',
                  data=df_sub[df_sub['ArrType'] == 'Fixed'],
                  hue='Array',
                  size=10,
                  alpha=0.75,
                  marker='o',
                  jitter=0.3,
                  palette='viridis',
                  legend=False,
                  zorder=1)
    sns.stripplot(ax=axes[row, col],
                  x='ArrayLoc',
                  y='PercRaft',
                  data=df_sub[df_sub['ArrType'] == 'Live'],
                  hue='Array',
                  size=10,
                  alpha=0.75,
                  marker='^',
                  jitter=0.3,
                  palette='viridis',
                  legend=False,
                  zorder=1)

    axes[row, col].errorbar(x=df_sub_mean['ArrayLoc'],
                            y=df_sub_mean['Mean'],
                            yerr=df_sub_mean['Stdev'],
                            fmt='none',
                            color='black',
                            capsize=15,
                            capthick=3,
                            linewidth=3,
                            zorder=2)
    for i, row2 in df_sub_mean.iterrows():
        axes[row, col].hlines(y=row2['Mean'],
                              xmin=(i - 0.4),
                              xmax=(i + 0.4),
                              colors='black',
                              linewidth=3,
                              zorder=3)

    axes[row, col].set_xlabel('')
    axes[row, col].set_xticks([0, 1],
                              arrLocs_plot,
                              fontweight='bold')
    axes[row, col].set_xlim(-0.5, 1.5)
    axes[row, col].set_ylabel(f'% {grad_labs[idx]}', fontweight='bold')
    max_ylim = np.max(df_sub['PercRaft'])
    axes[row, col].set_ylim(0, (max_ylim * 1.22))

    axes[row, col].plot([0, 0, 1, 1],
                        [max_ylim * 1.07,
                         max_ylim * 1.1,
                         max_ylim * 1.1,
                         max_ylim * 1.07],
                        linewidth=2,
                        color='k')
    if sig[idx] == 'n.s.':
        height_factor = 1.1
    else:
        height_factor = 1.06

    axes[row, col].text((0 + 1) * 0.5,
                        max_ylim * height_factor,
                        sig[idx],
                        ha='center',
                        va='bottom',
                        color='k',
                        fontsize=16
                        )

plt.tight_layout()
plt.show()

#%% [markdown]
# Patterning metrics

#%%
from pathlib import Path
import pandas as pd
import numpy as np
import scipy.stats as stats

import seaborn as sns
import matplotlib.pyplot as plt

#%%
dataDir = Path(r'C:\Users\MVX\OneDrive - UW\Allbritton Lab\Confocal\241126_Micropattern Staining_Raft')

#%%
arrays = [_ for _ in range(1, 4)]
arrLocs = ['Center', 'Edge', 'Total']
numCenRafts = (23 - (2 * 2)) ** 2
grads = ['Pass', 'Missing', 'Border']

#%%
df_counts = pd.read_csv(dataDir / 'Pattern Grading Count.csv')

arrLocs_col = df_counts['ArrayLoc'].tolist()
counts_col = df_counts['Count'].tolist()
percRafts = []

for idx, arrLoc in enumerate(arrLocs_col):
    count = counts_col[idx]
    if arrLoc == 'Center':
        percRafts.append(count / numCenRafts * 100)
    elif arrLoc == 'Edge':
        percRafts.append(count / (23**2 - numCenRafts) * 100)
    elif arrLoc == 'Total':
        percRafts.append(count / 23**2 * 100)
df_counts['PercRaft'] = percRafts

#%%
grads_col2 = []
arrLocs_col2 = []
means_col2 = []
stdevs_col2 = []

for grad in grads:
    for arrLoc in arrLocs:
        grads_col2.append(grad)
        arrLocs_col2.append(arrLoc)

        df_sub = df_counts[(df_counts['Grade'] == grad) &
                           (df_counts['ArrayLoc'] == arrLoc)]
        mean = np.mean(df_sub['PercRaft'])
        stdev = np.std(df_sub['PercRaft'])

        means_col2.append(mean)
        stdevs_col2.append(stdev)

df_counts_means = pd.DataFrame({'Grade': grads_col2,
                              'ArrayLoc': arrLocs_col2,
                              'Mean': means_col2,
                              'Stdev': stdevs_col2})

#%%
df_counts.to_csv((dataDir / 'Pattern Grading Count_Perc.csv'), index=False)
df_counts_means.to_csv((dataDir / 'Pattern Grading Count_Mean.csv'), index=False)

#%%
plt.rcdefaults()
sns.set_theme(style='white', font_scale=1.25)
plt.rcParams['ytick.left'] = True
fig, axes = plt.subplots(1, 1, figsize=(4, 4))

df_sub = df_counts[df_counts['ArrayLoc'] == 'Total']
df_sub.reset_index(drop=True, inplace=True)
df_sub_mean = df_counts_means[df_counts_means['ArrayLoc'] == 'Total']
df_sub_mean.reset_index(drop=True, inplace=True)

sns.swarmplot(ax=axes,
              x='Grade',
              y='PercRaft',
              data=df_sub,
              hue='Array',
              size=15,
              palette='colorblind',
              legend=False,
              zorder=1)
# axes.errorbar(x=df_sub_mean['Grade'],
#               y=df_sub_mean['Mean'],
#               yerr=df_sub_mean['Stdev'],
#               fmt='none',
#               color='black',
#               capsize=15,
#               capthick=3,
#               linewidth=3,
#               zorder=2)
# for i, row in df_sub_mean.iterrows():
#     axes.hlines(y=row['Mean'],
#                xmin= (i - 0.5),
#                xmax= (i + 0.5),
#                colors='black',
#                linewidth=3,
#                zorder=1)

axes.set_xlabel('')
axes.set_xticks([0, 1, 2],
                grads,
                fontweight='bold')
axes.set_ylabel('% Rafts', fontweight='bold')
axes.set_ylim(0, (np.max(df_sub['PercRaft']) * 1.1))

plt.tight_layout()
plt.show()

#%%
for grad in grads:
    df_sub = df_counts[df_counts['Grade'] == grad]
    df_center = df_sub[df_sub['ArrayLoc'] == 'Center']
    df_edge = df_sub[df_sub['ArrayLoc'] == 'Edge']
    ttest = stats.ttest_ind(a=df_center['PercRaft'],
                            b=df_edge['PercRaft'],
                            equal_var=True)
    print(f'{grad}: {ttest}')

#%%
arrLocs_plot = arrLocs[:-1]
sig = ['***', 'n.s.', '***']

plt.rcdefaults()
sns.set_theme(style='white', font_scale=1.25)
plt.rcParams['ytick.left'] = True
fig, axes = plt.subplots(1, 3, figsize=(10, 4))

for idx, grad in enumerate(grads):
    df_sub = df_counts[(df_counts['Grade'] == grad) &
                            (df_counts['ArrayLoc'] != 'Total')]
    df_sub.reset_index(drop=True, inplace=True)
    df_sub_mean = df_counts_means[(df_counts_means['Grade'] == grad) &
                                  (df_counts_means['ArrayLoc'] != 'Total')]
    df_sub_mean.reset_index(drop=True, inplace=True)

    sns.swarmplot(ax=axes[idx],
                  x='ArrayLoc',
                  y='PercRaft',
                  data=df_sub,
                  hue='Array',
                  size=15,
                  palette='colorblind',
                  legend=False,
                  zorder=1)

    # axes[idx].errorbar(x=df_sub_mean['ArrayLoc'],
    #                    y=df_sub_mean['Mean'],
    #                    yerr=df_sub_mean['Stdev'],
    #                    fmt='none',
    #                    color='black',
    #                    capsize=15,
    #                    capthick=3,
    #                    linewidth=3,
    #                    zorder=2)
    # for i, row2 in df_sub_mean.iterrows():
    #     axes[idx].hlines(y=row2['Mean'],
    #                      xmin=(i - 0.4),
    #                      xmax=(i + 0.4),
    #                      colors='black',
    #                      linewidth=3,
    #                      zorder=1)

    axes[idx].set_xlabel('')
    axes[idx].set_xticks([0, 1],
                              arrLocs_plot,
                              fontweight='bold')
    axes[idx].set_xlim(-0.5, 1.5)
    axes[idx].set_ylabel(f'% {grads[idx]}', fontweight='bold')
    max_ylim = np.max(df_sub['PercRaft'])

    if idx != 1:
        axes[idx].set_ylim(0, max_ylim * 1.20)

        axes[idx].plot([0, 0, 1, 1],
                       [max_ylim * 1.07,
                        max_ylim * 1.1,
                        max_ylim * 1.1,
                        max_ylim * 1.07],
                       linewidth=2,
                       color='k')
        if sig[idx] == 'n.s.':
            height_factor = 1.1
        else:
            height_factor = 1.07

        axes[idx].text((0 + 1) * 0.5,
                       max_ylim * height_factor,
                       sig[idx],
                       ha='center',
                       va='bottom',
                       color='k',
                       fontsize=16
                       )

axes[1].set_ylim(0, 1)
axes[1].yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
axes[1].plot([0, 0, 1, 1],
             [0.075,
              0.1,
              0.1,
              0.075],
             linewidth=2,
             color='k')
axes[1].text((0 + 1) * 0.5,
             0.1,
             sig[1],
             ha='center',
             va='bottom',
             color='k',
             fontsize=16
             )

plt.tight_layout()
plt.show()

#%%
df_diams = pd.read_csv(dataDir / 'Pattern Diameter.csv')

arrays = [_ for _ in range(1,4)]
arrLocs = ['Center', 'Edge']

#%%
arrays_col2 = []
arrLocs_col2 = []
means_col2 = []
stdevs_col2 = []

for array in arrays:
    for arrLoc in (arrLocs + ['Total']):
        arrays_col2.append(array)
        arrLocs_col2.append(arrLoc)

        if arrLoc != 'Total':
            df_sub = df_diams[(df_diams['Array'] == array) &
                              (df_diams['ArrayLoc'] == arrLoc)]
        else:
            df_sub = df_diams[df_diams['Array'] == array]
        diam = df_sub['Diameter'].tolist()
        means_col2.append(np.mean(diam))
        stdevs_col2.append(np.std(diam))

df_diams_means = pd.DataFrame({'Array': arrays_col2,
                               'ArrayLoc': arrLocs_col2,
                               'Mean': means_col2,
                               'Stdev': stdevs_col2})


#%%
arrLocs_col3 = []
means_col3 = []
stdevs_col3 = []

for arrLoc in (arrLocs + ['Total']):
    arrLocs_col3.append(arrLoc)

    df_sub = df_diams_means[df_diams_means['ArrayLoc'] == arrLoc]
    mean = df_sub['Mean'].tolist()
    means_col3.append(np.mean(mean))
    stdevs_col3.append(np.std(mean))

df_diams_means2 = pd.DataFrame({'ArrayLoc': arrLocs_col3,
                               'Mean': means_col3,
                               'Stdev': stdevs_col3})

#%%
df_diams_means.to_csv((dataDir / 'Pattern Diameter_Mean.csv'),
                      index = False)
df_diams_means2.to_csv((dataDir / 'Pattern Diameter_Mean2.csv'),
                      index = False)

#%%
import scipy.stats as stats

df_diam_center = df_diams_means[(df_diams_means['ArrayLoc'] == 'Center')]
df_diam_edge = df_diams_means[(df_diams_means['ArrayLoc'] == 'Edge')]

ttest_diam = stats.ttest_ind(a=df_diam_center['Mean'],
                             b=df_diam_edge['Mean'],
                             equal_var=True)
print(f'Pattern Diam: {ttest_diam}')

#%%%
plt.rcdefaults()
sns.set_theme(style='white', font_scale=1.25)
plt.rcParams['ytick.left'] = True
fig, axes = plt.subplots(1, 1, figsize=(4, 4))

sns.stripplot(ax=axes,
              x='ArrayLoc',
              y='Diameter',
              data=df_diams,
              hue='Array',
              size=8,
              palette='colorblind',
              alpha=0.5,
              jitter=0.2,
              legend=False,
              zorder=1)
sns.swarmplot(ax=axes,
              x='ArrayLoc',
              y='Mean',
              data=df_diams_means,
              hue='Array',
              size=15,
              palette='colorblind',
              edgecolor='k',
              linewidth=2,
              legend=False,
              zorder=3)
axes.set_xticks([0, 1], arrLocs, fontweight='bold')
axes.set_xlabel('', fontweight='bold')
axes.set_xlim(-0.5, 1.5)
axes.set_ylabel('Micropattern Diameter (μm)', fontweight='bold')
minMax = [min(df_diams['Diameter']), max(df_diams['Diameter'])]
axes.set_ylim(minMax[0] * 0.97,
              minMax[1] * 1.025)
axes.yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=4))

axes.plot([0, 0, 1, 1],
          [minMax[1] * 1.007,
           minMax[1] * 1.01,
           minMax[1] * 1.01,
           minMax[1] * 1.007],
          linewidth=2,
          color='k')
axes.text((0 + 1) * 0.5,
          minMax[1] * 1.01,
          "n.s.",
          ha='center',
          va='bottom',
          color='k',
          fontsize=16)

# axes.errorbar(x=df_diams_means2['ArrayLoc'],
#               y=df_diams_means2['Mean'],
#               yerr=df_diams_means2['Stdev'],
#               fmt='none',
#               color='black',
#               capsize=15,
#               capthick=3,
#               marker='_',
#               linewidth=3,
#               zorder=2)
# for i, row2 in df_diams_means2.iterrows():
#     axes.hlines(y=row2['Mean'],
#                 xmin=(i - 0.4),
#                 xmax=(i + 0.4),
#                 colors='black',
#                 linewidth=3,
#                 zorder=2)

plt.tight_layout()
plt.show()

#%%
df_fromCen = pd.read_csv(dataDir / 'Pattern From Center.csv')

arrays = [_ for _ in range(1, 4)]
arrLocs = ['Center', 'Edge']
patterns = [_ for _ in range(1,9)]

#%%
print(df_fromCen['Array'].unique())
# df_fromCen['Array'] = df_fromCen['Array'].astype(int)
# df_fromCen['Pattern'] = df_fromCen['Pattern'].astype(int)

#%%
cenDists_col = []
for array in arrays:
    for arrLoc in arrLocs:
        for patt in patterns:
            df_sub = df_fromCen[(df_fromCen['Array'] == array) &
                                (df_fromCen['ArrayLoc'] == arrLoc) &
                                (df_fromCen['Pattern'] == patt)]
            dists = df_sub['Distance'].tolist()
            cenDist = (np.max(dists) - np.min(dists)) / 2
            # print(array, arrLoc, patt, dists)
            cenDists_col.extend(cenDist for dist in dists)
df_fromCen['CenDist'] = cenDists_col

#%%
fromCenDist_col = []
dists = df_fromCen['Distance'].tolist()
for idx, dist in enumerate(dists):
    fromCenDist_col.append(abs(dist - cenDists_col[idx]))
df_fromCen['FromCenDist'] = fromCenDist_col

#%%
cenInts_col = []
for array in arrays:
    for arrLoc in arrLocs:
        for patt in patterns:
            df_sub = df_fromCen[(df_fromCen['Array'] == array) &
                                (df_fromCen['ArrayLoc'] == arrLoc) &
                                (df_fromCen['Pattern'] == patt)]
            df_sub2 = df_sub[df_sub['FromCenDist'] < 10]
            cenInt = np.mean(df_sub2['Intensity'].astype(float).tolist())
            # print(array, arrLoc, patt, cenInt)
            cenInts_col.extend([cenInt for _ in range(df_sub.shape[0])])
df_fromCen['CenInt'] = cenInts_col

#%%
normInts_col = []
ints = df_fromCen['Intensity'].tolist()
for idx, cenInt in enumerate(cenInts_col):
    normInts_col.append(ints[idx] / cenInt)
df_fromCen['NormInt'] = normInts_col

#%%
df_fromCen.to_csv((dataDir / 'Pattern From Center_Norm.csv'),
                  index = False)

#%%
plt.rcdefaults()
sns.set_theme(style='white', font_scale=1.25)
plt.rcParams['ytick.left'] = True
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

for idx, arrLoc in enumerate(arrLocs):
    df_sub = df_fromCen[df_fromCen['ArrayLoc'] == arrLoc]

    df_smooth = df_sub.sort_values(by='FromCenDist')
    df_smooth['NormInt_Smooth'] = df_smooth['NormInt'].rolling(50).mean()
    df_smooth.to_csv((dataDir / f'Pattern From Center_Smooth_{arrLoc}.csv'),
                     index=False)

    fromCenDists_smooth = df_smooth['FromCenDist'].tolist()
    normInts_smooth = df_smooth['NormInt_Smooth'].tolist()
    i = 0
    thresh = fromCenDists_smooth[0]
    while (normInts_smooth[i] >= 0.9) or (np.isnan(normInts_smooth[i])):
        thresh = fromCenDists_smooth[i]
        i += 1
    print(f'{arrLoc} <0.9: {thresh}')

    i = 0
    thresh = fromCenDists_smooth[0]
    while (normInts_smooth[i] >= 0.8) or (np.isnan(normInts_smooth[i])):
        thresh = fromCenDists_smooth[i]
        i += 1
    print(f'{arrLoc} <0.8: {thresh}')

    sns.scatterplot(ax=axes[idx],
                    x='FromCenDist',
                    y='NormInt',
                    data=df_sub,
                    hue='Array',
                    size=8,
                    palette='colorblind',
                    alpha=0.25,
                    legend=False,
                    zorder=2)
    sns.lineplot(ax=axes[idx],
                 x='FromCenDist',
                 y='NormInt_Smooth',
                 data=df_smooth,
                 errorbar='sd',
                 hue='Array',
                 palette='colorblind',
                 linewidth=6,
                 legend=False,
                 zorder=2)

    axes[idx].set_xlabel(f'Distance from Micropattern Center (μm)\n{arrLoc}', fontweight='bold')
    axes[idx].set_xlim(0, 290)
    axes[idx].set_ylabel('Normalized Intensity', fontweight='bold')
    # axes.yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=4))

plt.tight_layout()
plt.show()

#%%
# # Run only for initialization!
# dataDir = Path(r'C:\Users\MVX\OneDrive - UW\Allbritton Lab\Confocal\241126_Micropattern Staining_Raft')
# arrays = [_ for _ in range(1, 5)]
# arrLocs = ['Center', 'Edge']
# patts = [_ for _ in range(1, 9)]
#
# arrays_col = []
# arrLocs_col = []
# patts_col = []
# diams_col = []
#
# for array in arrays:
#     for arrLoc in arrLocs:
#         for patt in patts:
#             arrays_col.append(array)
#             arrLocs_col.append(arrLoc)
#             patts_col.append(patt)
#             diams_col.append(None)
#
# df_pattDiams = pd.DataFrame({'Array': arrays_col,
#                              'ArrayLoc': arrLocs_col,
#                              'Pattern': patts_col,
#                              'Diameter': diams_col})
#
# df_pattDiams.to_csv((dataDir / 'Pattern Diameter.csv'),
#                     index=False)

#%% [markdown]
# Master template dimensions

#%%
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#%%
dataDir = Path(r'C:\Users\MVX\OneDrive - UW\Allbritton Lab\Profilometer\210901_800umTemplate2')

df_height = pd.read_csv(dataDir / '210901_800umTemplate2_Height.csv')
df_width = pd.read_csv(dataDir / '210901_800umTemplate2_Width.csv')
df_gap = pd.read_csv(dataDir / '210901_800umTemplate2_Gap.csv')
df_temps = [df_height, df_width, df_gap]

measures = ['Height', 'Width', 'Gap']
temps = [_ for _ in range(1,6)]

#%%
df_means = []

for idx, df in enumerate(df_temps):
    temps_col = []
    means_col = []
    stdevs_col = []

    for temp in temps:
        df_sub = df[df['Template'] == temp]
        mean = np.mean(df_sub['Measurement'].tolist())
        stdev = np.std(df_sub['Measurement'].tolist())

        temps_col.append(temp)
        means_col.append(mean)
        stdevs_col.append(stdev)

    df_mean = pd.DataFrame({'Template': temps_col,
                             'Mean': means_col,
                             'Stdev': stdevs_col})
    df_means.append(df_mean)
    df_mean.to_csv((dataDir / f'210901_800umTemplate2_{measures[idx]}_Mean.csv'),
                   index=False)

#%%
means2_col = []
stdevs2_col = []

for idx, df in enumerate(df_means):
    mean2 = np.mean(df['Mean'].tolist())
    stdev2 = np.std(df['Mean'].tolist())

    means2_col.append(mean2)
    stdevs2_col.append(stdev2)

df_means2 = pd.DataFrame({'Measure': measures,
                         'Mean': means2_col,
                         'Stdev': stdevs2_col})
df_means2.to_csv((dataDir / '210901_800umTemplate2_Mean2.csv'),
                 index=False)

#%%
plt.rcdefaults()
sns.set_theme(style='white', font_scale=1.25)
plt.rcParams['ytick.left'] = True
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for idx, measure in enumerate(measures):
    df = df_temps[idx]
    df_mean = df_means[idx]

    sns.stripplot(ax=axes[idx],
                  y='Measurement',
                  data=df,
                  hue='Template',
                  size=10,
                  alpha=0.75,
                  jitter=0.3,
                  palette='colorblind',
                  legend=False,
                  zorder=1)
    sns.swarmplot(ax=axes[idx],
                  y='Mean',
                  data=df_mean,
                  hue='Template',
                  size=15,
                  palette='colorblind',
                  edgecolor='k',
                  linewidth=2,
                  legend=False,
                  zorder=3)

    axes[idx].errorbar(x=0,
                       y=means2_col[idx],
                       yerr=stdevs2_col[idx],
                       color='black',
                       capsize=20,
                       capthick=3,
                       linewidth=3,
                       marker='_',
                       markersize=125,
                       markeredgewidth=3,
                       zorder=2)

    axes[idx].set_xlabel('')
    axes[idx].set_ylabel(f'{measures[idx]} (μm)', fontweight='bold')
    axes[idx].yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=5))

plt.tight_layout()
plt.show()
