# stitchArray.py - Creates stitched composite image of
# entire array at each time point and for each channel.

from pathlib import Path
import os
from scipy.io import loadmat
import natsort

from PIL import Image
from matplotlib import pyplot as plt
from skimage import io
from skimage.transform import rescale
from skimage.filters import gaussian
from skimage.exposure import rescale_intensity
from skimage.util import img_as_ubyte
import numpy as np


def stitchArray(exp):
    # Load Positions.mat (real world coord of imaging frame centers) and initialize output path for stitched images.
    positionsPath = Path(exp.dataPath / 'data' / 'Positions.mat')
    positions = loadmat(positionsPath)
    outPath = Path(f'{exp.dataPath}_Analysis') / 'Stitched Images'
    os.makedirs(outPath, exist_ok=True)

    # Stitch images for each array at each time point and channel.
    # ch_stitch = ['BF']
    ch_stitch = ['BF', 'Violet']
    # ch_stitch = ['BF', 'Green']
    # ch_stitch = ['BF', 'Blue']
    # ch_stitch = ['BF', 'Blue', 'Green']

    orderTP = list(range(exp.numTimePoints))

    for arr in range(exp.numArrs):
        arrPath = exp.dataPath / f'{arr + 1}'
        posDict = img_pos(exp, arr, positions)
        for tp in orderTP:
            print(f'\tCurrently processing Array {arr + 1} at Time Point {tp}...')

            for ch in ch_stitch:
                imgPaths = []   # list of paths as strings
                for sub in os.listdir(arrPath):
                    imgPath = arrPath / sub / f'{tp:02}_{ch}.tif'
                    imgPaths.append(str(imgPath))
                imgPaths = natsort.natsorted(imgPaths)

                # print(f'Image paths for Array {arr + 1} at Time Point {tp} in {ch} channel: {imgPaths}')
                # print(f'There are {len(imgPaths)} images for '
                #       f'Array {arr + 1} at '
                #       f'Time Point {tp} in '
                #       f'{ch} channel.')
                # print(f'There are {len(imgPaths)} paths for this channel: {imgPaths}')

                stitchedImg = tile_stitch(exp, posDict, imgPaths, ch)

                stitchedImg.save(Path(outPath / f'array{arr + 1}_timepoint{tp:02}_{ch}.tif'),
                                 'TIFF')
                # stitchedImg.show()

                # print(f'Finished stitching images for Array {arr + 1} at '
                #       f'Time Point {tp} in '
                #       f'{ch} channel...')

    print(f'Stitched composites for '
          f'{exp.numArrs} array(s), '
          f'{exp.numTimePoints} Time Point(s), and '
          f'{len(ch_stitch)} Channel(s).')

    print('Stitch complete.')

# Create dictionary with xy locations in px for all imaging frames of each array from Positions.mat.
def img_pos(exp, arr, positions):
    pos = positions['Positions'][arr, 0]  # FOV positions for specific array
    numCols = len(pos[0])
    numRows = len(pos[:, 0])

    posDict = {}
    imgCount = 1
    for i in range(numRows):
        for j in range(numCols):
            posDict.setdefault(imgCount, {'x': 0, 'y': 0})
            posDict[imgCount]['x'] = pos[i][j][0] / exp.pxSize  # convert x coord from um to px
            posDict[imgCount]['y'] = pos[i][j][1] / exp.pxSize  # convert y coord from um to px
            imgCount += 1
    # print(posDict)
    return posDict


# Stitch the images within imgPaths based on image locations.
def tile_stitch(exp, posDict, imgPaths, ch):
    # print(f'Stitching images for Array {arr + 1} at '
    #       f'Time Point {tp} in '
    #       f'{ch} channel...')

    # Initialize parameters for tiled image.
    img = io.imread(imgPaths[0])
    tileSize = (int(img.shape[0] * exp.tileScale), int(img.shape[1] * exp.tileScale))

    distArr = (exp.arrDim[2] * exp.arrDim[0]) + ((exp.arrDim[2] - 1) * exp.arrDim[1])  # distance across array (um)
    outputSize = int(distArr / exp.pxSize * (exp.tileScale * 1.25))
    stitchedImg = np.zeros([outputSize, outputSize])
    images = []

    if ch == 'BF':
        sig = 15
    else:
        sig = 15

    # Resize and convert 16bit to 8bit images before storing in "images" list.
    # imgCount = 1
    for image in imgPaths:
        img_arr = io.imread(image)
        img_arr = rescale(img_arr, exp.tileScale)
        background = gaussian(img_arr, sigma=sig, preserve_range=True)
        img_arr = img_arr / background
        images.append(img_arr)

        # print('Currently on img %s' % counter)
        # imgCount += 1
    # print(f'Finished processing {imgCount - 1} images.')

    # Paste tiles into position with top-left image placed at (0, 0).
    imgCount = 1
    (x, y) = (0, 0)
    x_offset = posDict[1]['x'] * exp.tileScale
    y_offset = posDict[1]['y'] * exp.tileScale

    for img in images:
        (x, y) = (posDict[imgCount]['x'] * exp.tileScale, posDict[imgCount]['y'] * exp.tileScale)
        # print(f'Img {imgCount} is located at ({x - x_offset},{y - y_offset})')
        (new_x, new_y) = (int(x - x_offset), int(y - y_offset))
        if new_x < 0:
            new_x = 0
        if new_y < 0:
            new_y = 0
        # print(new_y, new_y+img.shape[0], new_x, new_x+img.shape[1])
        stitchedImg[new_y:new_y+img.shape[0], new_x:new_x+img.shape[1]] = img
        imgCount += 1

    # Crop stitched image to the bottom right of array before returning stitched image.
    right = x - x_offset + tileSize[0]
    bottom = y - y_offset + tileSize[1]
    stitchedImg = stitchedImg[0:int(bottom), 0:int(right)]

    background = gaussian(stitchedImg, sigma=30, preserve_range=True)
    stitchedImg = stitchedImg / background
    stitchedImg = rescale_intensity(stitchedImg)
    stitchedImg = img_as_ubyte(stitchedImg)
    stitchedImg = Image.fromarray(stitchedImg)
    # stitchedImg.show()

    return stitchedImg