import os
from pathlib import Path

DIR = Path(r'D:\2 - Data\230414_H9C2N1_1.2e6_46hrBMP4_ffc\1')
for sub in os.listdir(DIR):
    for i in range(2):
        for j in ['BF', 'Blue', 'Green']:
            imgPath = DIR / sub / f'{i:02}_{j}_ffc.tif'
            if os.path.exists(str(imgPath)):
                os.remove(imgPath)