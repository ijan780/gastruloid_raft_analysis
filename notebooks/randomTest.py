#%%
from pathlib import Path
import os
from scipy.io import loadmat

loc_cell2 = loadmat(Path(r'C:\Users\MVX\Documents\2 - Data\240805_R2GLR_1.8e6_0-48hrBMP4_0%aneu_4\data\Loc_cell2.mat'))
loc_cell2 = loc_cell2['Loc_cell2']
loc_cell2
#%%
raft_num = 0
raft_loc2 = [tp[raft_num] for tp in loc_cell2[1, :]]
raft_loc2

#%%
loc_idx = [tp[3] for tp in raft_loc2]
loc_idx

#%%
centroid_cell = loadmat(Path(r'C:\Users\MVX\Documents\2 - Data\240805_R2GLR_1.8e6_0-48hrBMP4_0%aneu_4\data\Centroid_cell.mat'))
centroid_cell = centroid_cell['Centroid_cell']
centroid_cell[1,0][16,16].shape[0]

#%%
import tifffile as tiff
from skimage import io
from skimage.exposure import rescale_intensity, equalize_adapthist
from skimage.morphology import disk, closing, opening, remove_small_holes, remove_small_objects
from skimage.filters import threshold_otsu, gaussian
from skimage.measure import label, regionprops
import numpy as np
import matplotlib.pyplot as plt

img_path = r'C:\Users\MVX\Documents\2 - Data\Analysis\240722_R2GLR_1.8e6_0-48hrBMP4_0%aneu_2_Analysis\Individual Raft Images\Time-lapse\array1_raft319_tl.tif'
with tiff.TiffFile(img_path) as tif:
    img = tif.pages[3].asarray()
io.imshow(img)
plt.show()

gaus1 = gaussian (img, sigma=4)
io.imshow(gaus1)
plt.show()

# img = equalize_adapthist(gaus1)
# io.imshow(img)
# plt.show()

gaus2 = gaussian(img, sigma=10)
io.imshow(gaus2)
plt.show()

#%%
ffc = gaus1 / gaus2
ffc2 = rescale_intensity(ffc,
                         in_range=(np.quantile(ffc, 0.05),
                                   np.quantile(ffc, 0.95))
                         )
io.imshow(ffc2)
plt.show()

#%%
# ffc3 = gaussian(ffc, sigma=4)

threshVal = threshold_otsu(ffc)
thresh = ffc > threshVal
print(threshVal)
io.imshow(thresh)
plt.show()

#%%
thresh = opening(thresh, disk(5))
thresh = remove_small_holes(thresh, area_threshold=5e4)
io.imshow(thresh)
plt.show()

labels = label(thresh)
regions = regionprops(labels)
largest_region = max(regions, key=lambda r: r.area)
largest_region_mask = np.zeros_like(labels, dtype=bool)
largest_region_mask[labels == largest_region.label] = True

# io.imshow(largest_region_mask)

# io.imshow(thresh)

#%%
list = [((tp * 6)) for _ in range(1, (529 + 1)) for tp in range(9)]

print(list)

#%%
from pathlib import Path

import tifffile as tiff

from skimage.transform import rescale
from skimage.filters import threshold_li, gaussian
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.io import imshow
import matplotlib.pyplot as plt
import numpy as np

main_dir = r'C:\Users\MVX\OneDrive - UW\Documents - gastruloid-raft\T2-T6_Gastruloids_T1'
samp_name = 'T6'
channel_centroid = 'GATA3'
img_dir = Path(main_dir) / f'{samp_name}_{channel_centroid}.tif'

with tiff.TiffFile(img_dir) as tif:
    img = tif.asarray()

#%%
from skimage.filters import threshold_li, threshold_otsu, gaussian

rescale_factor = 0.01
img_small = rescale(img, scale=rescale_factor)
imshow(img_small)
plt.show()

gaus = gaussian(img_small, sigma=4)
imshow(gaus)
plt.show()

thresh = gaus > threshold_otsu(gaus)
imshow(thresh)
plt.show()

clean = clear_border(thresh)
imshow(clean)
plt.show()

labels = label(clean)
regions = regionprops(labels)

#%%
from skimage.exposure import rescale_intensity
from tqdm import tqdm

channels = ['GATA3', 'BRA', 'SOX2', 'DAPI']
channel_centroid = 'GATA3'

main_outDir = Path(main_dir) / samp_name
main_outDir.mkdir(exist_ok=True)

gastruloid_nums = []

padding = 10 # px

for ch in channels:
    if ch is not channel_centroid:
        img_dir = Path(main_dir) / f'{samp_name}_{ch}.tif'
        with tiff.TiffFile(img_dir) as tif:
            img = tif.asarray()

    pb = tqdm(regions, desc=f'Channel {ch}')
    for idx, region in enumerate(pb):
        centroid = np.round(region.centroid).astype(int)
        bottom = (centroid[0] - padding) * (1/rescale_factor)
        top = (centroid[0] + padding) * (1/rescale_factor)
        left = (centroid[1] - padding) * (1/rescale_factor)
        right = (centroid[1] + padding) * (1/rescale_factor)

        crop = img[int(bottom):int(top), int(left):int(right)]
        # crop_norm = rescale_intensity(crop,
        #                               in_range=(np.quantile(crop, 0.01),
        #                                         np.quantile(crop, 0.99))
        #                               )
        # imshow(crop_norm)
        # plt.show()
        #
        # print((idx + 1), centroid, crop.shape)

        if 0 not in crop.shape:
            gastruloid_nums.append(idx)
            crop_outDir = main_outDir / f'{samp_name}_{ch}_gastruloid{(idx + 1):03}.tif'
            tiff.imwrite(crop_outDir, crop)

pb.close()