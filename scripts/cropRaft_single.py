from pathlib import Path
import os
from scipy.io import loadmat

from skimage import io, filters, exposure, util
import numpy as np
from PIL import Image

dataPath = r'C:\Users\MVX\Documents\2 - Data\231019_RUES2_1.8e6_48hrBMP4_0%aneu_3'
arr = 0
tp = 0
ch = 'BF'
raft = 56
arrDim = [800, 100, 23]
pxSize = 1.3163

loc_cell2 = loadmat(Path(dataPath) / 'data' / 'Loc_cell2.mat')
loc_cell2 = loc_cell2['Loc_cell2']
centroid_cell = loadmat(Path(dataPath) / 'data' / 'Centroid_cell.mat')
centroid_cell = centroid_cell['Centroid_cell']

img_row = loc_cell2[arr, tp][raft, 1] - 1
img_col = loc_cell2[arr, tp][raft, 2] - 1
loc_idx = loc_cell2[arr, tp][raft, 3] - 1
# img_row = 6
# img_col = 2

raft_loc_x = centroid_cell[arr, tp][img_row, img_col][loc_idx][0]
raft_loc_y = centroid_cell[arr, tp][img_row, img_col][loc_idx][1]
# raft_loc_x = 1482
# raft_loc_y = 558

imgPath = Path(dataPath) / \
          f'{arr + 1}' / \
          f'{img_row + 1}x{img_col + 1}' / \
          f'{tp:02}_{ch}.tif'
img = io.imread(imgPath)

# Crop image to specific raft.
scale = 1.25
top = raft_loc_y - (scale * (arrDim[0] / 2) / pxSize)
bottom = raft_loc_y + (scale * (arrDim[0] / 2) / pxSize)
left = raft_loc_x - (scale * (arrDim[0] / 2) / pxSize)
right = raft_loc_x + (scale * (arrDim[0] / 2) / pxSize)

raftCrop = img[int(top):int(bottom), int(left):int(right)]
raftCrop = Image.fromarray(raftCrop)
raftCrop.show()
raftCrop = raftCrop.save(r'C:\Users\MVX\Desktop\Border.png')