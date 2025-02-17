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
dataDir = Path(r'') # data directory
arrs = [1, 2, 3]
locs = ['center', 'edge'] # array locations
px_size = 1.59170008 # um/px

arrs_col = []
locs_col = []
rafts_col = []
raftAreas_col = []
raftCens_col = []
beadAreas_col = []
normBeads_col = []
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
                if loc == 'center':
                    locs_col.append('Middle')
                elif loc == 'edge':
                    locs_col.append('Peripheral')
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
                normBead = beadArea / raftArea
                normBeads_col.append(normBead)

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
                         'NormBead': normBeads_col,
                         'FromEdge_Mean': fromEdges_mean_col,
                         'FromEdge_Stdev': fromEdges_stdev_col
                         })

df_beads.to_csv((dataDir / 'Bead Distribution.csv'),
                index=False)

#%%
arrs_col2 = []
locs_col2 = []
rafts_col2 = []
normBeads_mean_col2 = []
normBeads_stdev_col2 = []
fromEdges_mean2_col2 = []
fromEdges_stdev2_col2 = []

locs_lab = ['Middle', 'Peripheral']

for arr in arrs:
    for loc in locs_lab:
        df_sub = df_beads[(df_beads['Array'] == arr) &
                          (df_beads['Location'] == loc)
                          ]
        normBead_mean = np.mean(df_sub['NormBead'].tolist())
        normBead_stdev = np.std(df_sub['NormBead'].tolist())
        fromEdge_mean2 = np.mean(df_sub['FromEdge_Mean'].tolist())
        fromEdge_stdev2 = np.std(df_sub['FromEdge_Mean'].tolist())

        arrs_col2.append(arr)
        locs_col2.append(loc)
        normBeads_mean_col2.append(normBead_mean)
        normBeads_stdev_col2.append(normBead_stdev)
        fromEdges_mean2_col2.append(fromEdge_mean2)
        fromEdges_stdev2_col2.append(fromEdge_stdev2)

df_beads_mean = pd.DataFrame({'Array': arrs_col2,
                              'Location': locs_col2,
                              'NormBeads_Mean': normBeads_mean_col2,
                              'NormBeads_Stdev': normBeads_stdev_col2,
                              'FromEdge_Mean': fromEdges_mean2_col2,
                              'FromEdge_Stdev': fromEdges_stdev2_col2
                              })

df_beads_mean.to_csv((dataDir / 'Bead Distribution_Mean.csv'),
                index=False)

#%%
dataDir = Path(r'C:\Users\MVX\OneDrive - UW\Allbritton Lab\DIC\3. 800x800x100um\241029_Beads_Compiled')
df_beads = pd.read_csv(dataDir / 'Bead Distribution.csv')
df_beads_mean = pd.read_csv(dataDir / 'Bead Distribution_Mean.csv')