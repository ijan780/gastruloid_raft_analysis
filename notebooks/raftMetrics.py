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
import scipy.stats as stats

ttest_beadDens = stats.ttest_ind(a=df_beads_mean[(df_beads_mean['Location'] == 'center')
                                                     & (df_beads_mean['Location'] == 'center')
                                                     ]['BeadDens_Mean'],
                                     b=df_beads_mean[(df_beads_mean['Location'] == 'edge')
                                                     & (df_beads_mean['Location'] == 'edge')
                                                     ]['BeadDens_Mean'],
                                     equal_var=True)
print(f'Bead Area Dens: {ttest_beadDens}')

ttest_fromEdge = stats.ttest_ind(a=df_beads_mean[(df_beads_mean['Location'] == 'center')
                                                 & (df_beads_mean['Location'] == 'center')
                                                 ]['FromEdge_Mean'],
                                 b=df_beads_mean[(df_beads_mean['Location'] == 'edge')
                                                 & (df_beads_mean['Location'] == 'edge')
                                                 ]['FromEdge_Mean'],
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
              linewidth=1,
              legend=False,
              zorder=2)
axes[0].set_xticks([0, 1], ['Center', 'Edge'], fontweight='bold')
axes[0].set_xlabel('', fontweight='bold')
axes[0].set_xlim(-0.5,1.5)
axes[0].set_ylabel('Bead Density', fontweight='bold')
axes[0].set_ylim(min(df_beads['BeadDens']) * 0.9,
                 max(df_beads['BeadDens']) * 1.12)
axes[0].yaxis.set_major_locator(plt.MaxNLocator(nbins=4))

axes[0].plot([0, 0, 1, 1], 
             [max(df_beads['BeadDens']) + 1e-3,
              max(df_beads['BeadDens']) + 1.5e-3,
              max(df_beads['BeadDens']) + 1.5e-3,
              max(df_beads['BeadDens']) + 1e-3],
             linewidth=2, 
             color='k')
axes[0].text((0+1)*0.5, 
             max(df_beads['BeadDens']) + 1.5e-3,
             "n.s.",
             ha='center', 
             va='bottom', 
             color='k',
             fontsize=16
            )

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
              linewidth=1,
              legend=False,
              zorder=2)
axes[1].set_xticks([0, 1], ['Center', 'Edge'], fontweight='bold')
axes[1].set_xlabel('', fontweight='bold')
axes[1].set_xlim(-0.5, 1.5)
axes[1].set_ylabel('Distance from Edge (μm)', fontweight='bold')
axes[1].set_ylim(min(df_beads['FromEdge_Mean']) * 0.9,
                 max(df_beads['FromEdge_Mean']) * 1.1)
axes[1].yaxis.set_major_locator(plt.MaxNLocator(nbins=4))

axes[1].plot([0, 0, 1, 1],
             [max(df_beads['FromEdge_Mean']) + 0.5,
              max(df_beads['FromEdge_Mean']) + 1,
              max(df_beads['FromEdge_Mean']) + 1,
              max(df_beads['FromEdge_Mean']) + 0.5],
             linewidth=2,
             color='k')
axes[1].text((0+1)*0.5,
             max(df_beads['FromEdge_Mean']) + 1,
             "n.s.",
             ha='center',
             va='bottom',
             color='k',
             fontsize=16
            )

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
    for arr_loc in arr_locs:
        arrs_edge_col2.append(arr)
        arr_locs_edge_col2.append(arr_loc)

        df_sub = df_edges_means[(df_edges_means['Array'] == arr) &
                                (df_edges_means['ArrayLoc'] == arr_loc)]
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
    for arr_loc in arr_locs:
        for raft_loc in raft_locs:
            arrs_thick_col2.append(arr)
            arr_locs_thick_col2.append(arr_loc)
            raft_locs_thick_col2.append(raft_loc)

            df_sub = df_thicks_means[(df_thicks_means['Array'] == arr) &
                                     (df_thicks_means['ArrayLoc'] == arr_loc) &
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
import scipy.stats as stats

arrCenters_edge = df_edges_means2[df_edges_means2['ArrayLoc'] == 'Center']['Mean'].tolist()
arrEdges_edge = df_edges_means2[df_edges_means2['ArrayLoc'] == 'Edge']['Mean'].tolist()
ttest_edge = stats.ttest_ind(a=arrCenters_edge, b=arrEdges_edge, equal_var=False)
print(f'Raft Edges: {ttest_edge}')

center_arrCenters_thick = df_thicks_means2[(df_thicks_means2['ArrayLoc'] == 'Center') &
                                           (df_thicks_means2['RaftLoc'] == 'Center')]['Mean'].tolist()
center_arrEdges_thick = df_thicks_means2[(df_thicks_means2['ArrayLoc'] == 'Edge') &
                                         (df_thicks_means2['RaftLoc'] == 'Center')]['Mean'].tolist()
ttest_center_thick = stats.ttest_ind(a=center_arrCenters_thick, b=center_arrEdges_thick, equal_var=False)
print(f'Raft Center Thickness: {ttest_center_thick}')

edge_arrCenters_thick = df_thicks_means2[(df_thicks_means2['ArrayLoc'] == 'Center') &
                                         (df_thicks_means2['RaftLoc'] == 'Edge')]['Mean'].tolist()
edge_arrEdges_thick = df_thicks_means2[(df_thicks_means2['ArrayLoc'] == 'Edge') &
                                       (df_thicks_means2['RaftLoc'] == 'Edge')]['Mean'].tolist()
ttest_edge_thick = stats.ttest_ind(a=edge_arrCenters_thick, b=edge_arrEdges_thick, equal_var=False)
print(f'Raft Edge Thickness: {ttest_edge_thick}')

#%%
df_edges_means2.to_csv((dataDir / '800x800x104um Raft Measurements_Mean_Edge_Means.csv'),
                      index=False)
df_thicks_means2.to_csv((dataDir / '800x800x104um Raft Measurements_Mean_Thickness_Means.csv'),
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
              linewidth=1,
              legend=False,
              zorder=2)
axes.set_xticks([0, 1], arr_locs, fontweight='bold')
axes.set_xlabel('', fontweight='bold')
axes.set_xlim(-0.5, 1.5)
axes.set_ylabel('Raft Edge Length (μm)', fontweight='bold')
axes.set_ylim(min(df_edges_means['Measurement']) * 0.998,
              max(df_edges_means['Measurement']) * 1.004)
axes.yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=4))

axes.plot([0, 0, 1, 1],
          [max(df_edges_means['Measurement']) + 1,
           max(df_edges_means['Measurement']) + 1.5,
           max(df_edges_means['Measurement']) + 1.5,
           max(df_edges_means['Measurement']) + 1],
          linewidth=2,
          color='k')
axes.text((0 + 1) * 0.5,
          max(df_edges_means['Measurement']) + 1.5,
          "n.s.",
          ha='center',
          va='bottom',
          color='k',
          fontsize=16
          )

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
              linewidth=1,
              legend=False,
              zorder=2)
axes[0].set_xticks([0, 1], arr_locs, fontweight='bold')
axes[0].set_xlabel('', fontweight='bold')
axes[0].set_xlim(-0.5,1.5)
axes[0].set_ylabel('Raft Center Thickness (μm)', fontweight='bold')
axes[0].set_ylim(min(df_raftCen['Measurement']) * 0.96,
                 max(df_raftCen['Measurement']) * 1.06)
axes[0].yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=4))

axes[0].plot([0, 0, 1, 1],
          [max(df_raftCen['Measurement']) + 1.5,
           max(df_raftCen['Measurement']) + 2,
           max(df_raftCen['Measurement']) + 2,
           max(df_raftCen['Measurement']) + 1.5],
          linewidth=2,
          color='k')
axes[0].text((0 + 1) * 0.5,
          max(df_raftCen['Measurement']) + 2,
          "n.s.",
          ha='center',
          va='bottom',
          color='k',
          fontsize=16
          )

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
              linewidth=1,
              legend=False,
              zorder=2)
axes[1].set_xticks([0, 1], arr_locs, fontweight='bold')
axes[1].set_xlabel('', fontweight='bold')
axes[1].set_xlim(-0.5,1.5)
axes[1].set_ylabel('Raft Edge Thickness (μm)', fontweight='bold')
axes[1].set_ylim(min(df_raftEdge['Measurement']) * 0.98,
                 max(df_raftEdge['Measurement']) * 1.03)
axes[1].yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=4))

axes[1].plot([0, 0, 1, 1],
          [max(df_raftEdge['Measurement']) + 1.5,
           max(df_raftEdge['Measurement']) + 2,
           max(df_raftEdge['Measurement']) + 2,
           max(df_raftEdge['Measurement']) + 1.5],
          linewidth=2,
          color='k')
axes[1].text((0 + 1) * 0.5,
          max(df_raftEdge['Measurement']) + 2,
          "n.s.",
          ha='center',
          va='bottom',
          color='k',
          fontsize=16
          )

plt.tight_layout()
plt.show()
