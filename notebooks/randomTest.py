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
from skimage.restoration import denoise_bilateral
from skimage.measure import label, regionprops
import numpy as np
import matplotlib.pyplot as plt

img_path = r'C:\Users\MVX\Documents\2 - Data\Analysis\240805_R2GLR_1.8e6_0-48hrBMP4_0%aneu_4_Analysis\Individual Raft Images\Time-lapse\array1_raft052_tl.tif'
with tiff.TiffFile(img_path) as tif:
    img = tif.pages[1].asarray()
io.imshow(img)
plt.show()

img = equalize_adapthist(img) 
io.imshow(img)
plt.show()

# img = gaussian(img, sigma=2)
img = denoise_bilateral(img, sigma_spatial=2)
io.imshow(img)
plt.show()
tiff.imwrite(r'C:\Users\MVX\Documents\2 - Data\Analysis\240805_R2GLR_1.8e6_0-48hrBMP4_0%aneu_4_Analysis\Individual Raft Images\Test\array1_raft_052_tp0_denoise.tif',
             img)

threshVal = threshold_otsu(img)
thresh = img > threshVal
print(threshVal)
# io.imshow(thresh)

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