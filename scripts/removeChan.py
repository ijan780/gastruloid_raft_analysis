# Remove excess channels.
from pathlib import Path
import tifffile as tiff

#%%
imgDir = Path(r"C:\Users\MVX\OneDrive - UW\Allbritton Lab\Manuscripts\Manuscript 2\Data\DAGOMR Images\Raw\0609")
outDir = Path(r"C:\Users\MVX\OneDrive - UW\Allbritton Lab\Manuscripts\Manuscript 2\Data\DAGOMR Images\Raw\0609_Trunc")

#%%
for imgPath in imgDir.iterdir():
    with tiff.TiffFile(imgPath) as tif:
        img = tif.asarray() # TCYX

    img_2ch = img[:, :2, :, :]

    tiff.imwrite(
        (outDir / imgPath.name),
        img_2ch,
        imagej=True,
        metadata={'axes': 'TCYX',
                  'channels': ['BF', 'SOX2']}
                 )