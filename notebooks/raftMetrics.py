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
from mpl_toolkits.axes_grid1 import make_axes_locatable

dataDir = Path(r'C:\Users\MVX\OneDrive - UW\Allbritton Lab\Manuscripts\Manuscript 1 - Technical\Data\Bead Distribution')
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
fromEdge_means_col = []
fromEdge_stdev_col = []

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
            plt.imshow(mask_bf, cmap='gray', interpolation='nearest')
            plt.axis('off')
            plt.show()

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
                plt.imshow(mask_raft, cmap='gray', interpolation='nearest')
                plt.axis('off')
                plt.show()

                img_tx_masked = np.logical_and(mask_raft, mask_tx)
                plt.imshow(img_tx_masked, cmap='gray', interpolation='nearest')
                plt.axis('off')
                plt.show()

                # Bead area & density.
                beadArea = np.sum(img_tx_masked)
                beadAreas_col.append(beadArea)
                beadDens = beadArea / raftArea
                beadDens_col.append(beadDens)

                # Bead distribution.

                # Distance map for raft edges.
                mask_raft_pad = np.pad(mask_raft, pad_width=5, mode='constant', constant_values=0)
                distance_to_edge = distance_transform_edt(mask_raft_pad) # distance-to-edge map for raft mask

                plt.figure(figsize=(6, 6))
                ax = plt.gca()
                im = ax.imshow(distance_to_edge, cmap='viridis', interpolation='nearest')
                plt.axis('off')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                plt.colorbar(im, cax=cax, label='Distance to Nearest Edge (px)')
                plt.tight_layout()
                plt.show()

                img_tx_masked_pad = np.pad(img_tx_masked, pad_width=5, mode='constant', constant_values=0)
                non_zero_coords = np.argwhere(img_tx_masked_pad > 0)  # coord of all non-zero px
                distances = [distance_to_edge[y,x] for y, x in non_zero_coords]

                fromEdge_mean = (np.mean(distances) * px_size) / 800 # relative to raft dim
                fromEdge_means_col.append(fromEdge_mean)
                fromEdge_stdev = (np.std(distances) * px_size) / 800
                fromEdge_stdev_col.append(fromEdge_stdev)

df_beads = pd.DataFrame({'Array': arrs_col,
                         'Location': locs_col,
                         'Raft': rafts_col,
                         'RaftArea': raftAreas_col,
                         'RaftCen': raftCens_col,
                         'BeadArea': beadAreas_col,
                         'BeadDens': beadDens_col,
                         'FromEdgeMean': fromEdge_means_col,
                         'FromEdgeStdev': fromEdge_stdev_col
                         })

df_beads.to_csv((dataDir.parent / 'Bead Distribution.csv'),
                index=False)

#%%
arrs_col2 = []
locs_col2 = []
rafts_col2 = []
beadDens_mean_col2 = []
beadDens_sem_col2 = []
fromEdge_mean_col2 = []
fromEdge_sem_col2 = []

for arr in arrs:
    for loc in locs:
        df_sub = df_beads[(df_beads['Array'] == arr) &
                          (df_beads['Location'] == loc)
                          ]
        beadDens_mean = np.mean(df_sub['BeadDens'].tolist())
        beadDens_sem = np.std(df_sub['BeadDens'].tolist())
        fromEdge_rel_mean = np.mean(df_sub['FromEdgeMean'].tolist())
        fromEdge_rel_sem = np.std(df_sub['FromEdgeMean'].tolist())

        arrs_col2.append(arr)
        locs_col2.append(loc)
        beadDens_mean_col2.append(beadDens_mean)
        beadDens_sem_col2.append(beadDens_sem)
        fromEdge_mean_col2.append(fromEdge_rel_mean)
        fromEdge_sem_col2.append(fromEdge_rel_sem)

df_beads_mean = pd.DataFrame({'Array': arrs_col2,
                              'Location': locs_col2,
                              'BeadDens_Mean': beadDens_mean_col2,
                              'BeadDens_SEM': beadDens_sem_col2,
                              'FromEdge_Mean': fromEdge_mean_col2,
                              'FromEdge_SEM': fromEdge_sem_col2
                              })

outDir_beadAnal = Path(r'C:\Users\MVX\OneDrive - UW\Allbritton Lab\Manuscripts\Manuscript 1 - Technical\Data')
df_beads.to_csv((dataDir.parent / 'Bead Distribution_Means.csv'),
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
from matplotlib.ticker import FormatStrFormatter

plt.rcdefaults()
sns.set_theme(style='white', font_scale=1.25)
plt.rcParams['ytick.left'] = True
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

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
                 max(df_beads['BeadDens']) * 1.1)

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
              y='FromEdgeMean',
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
axes[1].set_ylabel('Realtive Distance from Edge', fontweight='bold')
axes[1].set_ylim(min(df_beads['FromEdgeMean']) * 0.9,
                 max(df_beads['FromEdgeMean']) * 1.1)

axes[1].plot([0, 0, 1, 1],
             [max(df_beads['FromEdgeMean']) + 1e-3,
              max(df_beads['FromEdgeMean']) + 1.5e-3,
              max(df_beads['FromEdgeMean']) + 1.5e-3,
              max(df_beads['FromEdgeMean']) + 1e-3],
             linewidth=2,
             color='k')
axes[1].text((0+1)*0.5,
             max(df_beads['FromEdgeMean']) + 1.5e-3,
             "n.s.",
             ha='center',
             va='bottom',
             color='k',
             fontsize=16
            )

plt.tight_layout()
plt.show()