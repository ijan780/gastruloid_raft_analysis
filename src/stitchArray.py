# stitchArray.py - Creates stitched composite image of
# entire array at each time point and for each channel.

from pathlib import Path
import os
from scipy.io import loadmat
import natsort

import tifffile as tiff
from skimage.transform import rescale
from skimage.filters import gaussian
from skimage.exposure import rescale_intensity
from skimage.util import img_as_ubyte
import numpy as np


def stitchArray(exp):
    # Create output folder for all stitched images.
    outDir_main = exp.analDir / 'Stitched Images'
    outDir_main.mkdir(exist_ok=True)

    # Define channels other than mask for stitching.
    ch_stitch = ['BF']

    # Load Positions.mat (real world coord of imaging frame centers) and initialize output dir for stitched images.
    posDir_mask = exp.endDir / 'data' / 'Positions.mat'
    positions_mask = loadmat(posDir_mask)
    posDir = exp.dataDir / 'data' / 'Positions.mat'
    positions = loadmat(posDir)
    orderTP = [0, (exp.numTimePoints - 1)] # just first and last time points
    # orderTP = list(range(exp.numTimePoints))

    # Determine time point of mask.
    if exp.dataDir == exp.endDir:
        maskTP = exp.numTimePoints - 1
    else:
        maskTP = 0

    for arr in range(exp.numArrs):
        # Stitch mask channel first.
        print(f'\tCurrently processing Array {arr + 1}')
        print(f'\t\tEndpoint - Channel {exp.channel_mask}')
        posDict_mask = img_pos(exp, arr, positions_mask)
        maskDirs = []
        arrDir_mask = exp.endDir / f'{arr + 1}'
        for sub in os.listdir(arrDir_mask):
            maskDir = arrDir_mask / sub / f'{maskTP:02}_{exp.channel_mask}.tif'
            maskDirs.append(maskDir)
        maskDirs = natsort.natsorted(maskDirs)
        stitchedImg = tile_stitch(exp, posDict_mask, maskDirs, exp.channel_mask)
        outDir = outDir_main / f'array{arr + 1}_timepoint{(exp.numTimePoints - 1):02}_{exp.channel_mask}.tif'
        tiff.imwrite(outDir, stitchedImg)

        # Stitch other channels next
        posDict = img_pos(exp, arr, positions)
        arrDir = exp.dataDir / f'{arr + 1}'
        for tp in orderTP:
            for ch in ch_stitch:
                print(f'\t\tTime Point {tp:02} - Channel {ch}')
                imgDirs = []
                for sub in os.listdir(arrDir):
                    imgDir = arrDir / sub / f'{tp:02}_{ch}.tif'
                    imgDirs.append(imgDir)
                imgDirs = natsort.natsorted(imgDirs)

                # print(f'Image paths for Array {arr + 1} at Time Point {tp} in {ch} channel: {imgDirs}')
                # print(f'There are {len(imgDirs)} images for '
                #       f'Array {arr + 1} at '
                #       f'Time Point {tp} in '
                #       f'{ch} channel.')
                # print(f'There are {len(imgDirs)} paths for this channel: {imgDirs}')

                stitchedImg = tile_stitch(exp, posDict, imgDirs, ch)
                outDir = outDir_main / f'array{arr + 1}_timepoint{tp:02}_{ch}.tif'
                tiff.imwrite(outDir, stitchedImg)

                # print(f'Finished stitching images for Array {arr + 1} at '
                #       f'Time Point {tp} in '
                #       f'{ch} channel...')

    print(f'Stitched composites for '
          f'{exp.numArrs} array(s), '
          f'{exp.numTimePoints} Time Point(s), and '
          f'{len(ch_stitch) + 1} Channel(s).')

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


# Stitch the images within imgDirs based on image locations.
def tile_stitch(exp, posDict, dirs, ch):
    # print(f'Stitching images for Array {arr + 1} at '
    #       f'Time Point {tp} in '
    #       f'{ch} channel...')

    # Initialize parameters for tiled image.
    with tiff.TiffFile(dirs[0]) as tif:
        img = tif.asarray()
        tileSize = (int(img.shape[0] * exp.tileScale), int(img.shape[1] * exp.tileScale))

    distArr = (exp.arrDim[2] * exp.arrDim[0]) + ((exp.arrDim[2] - 1) * exp.arrDim[1])  # distance across array (um)
    outputSize = int(distArr / exp.pxSize * (exp.tileScale * 1.25))
    stitchedImg = np.zeros([outputSize, outputSize])

    # Paste tiles into position with top-left image placed at (0, 0).
    (x, y) = (0, 0)
    x_offset = posDict[1]['x'] * exp.tileScale
    y_offset = posDict[1]['y'] * exp.tileScale

    # Set sigma values for gaussian blur.
    sig = 10

    # Resize and convert 16bit to 8bit images before storing in "images" list.
    # imgCount = 1
    for idx, image in enumerate(dirs):
        # Read each array section image.
        with tiff.TiffFile(image) as tif:
            img = tif.asarray()
            img = rescale(img, exp.tileScale)

            background = gaussian(img, sigma=sig, preserve_range=True)
            epsilon = 1e-10  # avoid dividing by 0
            img_ff = img / (background + epsilon)
            img_ff = rescale_intensity(img_ff,
                                       in_range=(np.quantile(img_ff, q=0.05), np.quantile(img_ff, q=0.95)))

        # Determine coordinates of array section.
        (x, y) = (posDict[idx + 1]['x'] * exp.tileScale, posDict[idx + 1]['y'] * exp.tileScale)
        # print(f'Img {idx + 1} is located at ({x - x_offset},{y - y_offset})')
        (new_x, new_y) = (max(0, int(x - x_offset)), max(0, int(y - y_offset)))
        # print(new_y, new_y+img.shape[0], new_x, new_x+img.shape[1])
        stitchedImg[new_y:new_y + img.shape[0], new_x:new_x + img.shape[1]] = img_ff

    # print(f'Finished processing {idx + 1} images.')

    # Crop stitched image to the bottom right of array before returning rescaled 8bit stitched image.
    right = x - x_offset + tileSize[0]
    bottom = y - y_offset + tileSize[1]
    stitchedImg = stitchedImg[0:int(bottom), 0:int(right)]
    stitchedImg = rescale_intensity(stitchedImg,
                                    in_range=(np.quantile(stitchedImg, q=0.05), np.quantile(stitchedImg, q=0.95)),)
    stitchedImg = img_as_ubyte(stitchedImg)

    return stitchedImg