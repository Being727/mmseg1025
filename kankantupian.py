import os

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt

img_path = '/home/niu/mmsegmentation/WHUDataset/img_dir/train/global_monthly_2018_02_mosaic_L15-1439E-1134N_5759_3655_13_1.jpg'
mask_path = '/home/niu/mmsegmentation/WHUDataset/ann_dir/train/global_monthly_2018_02_mosaic_L15-1439E-1134N_5759_3655_13_1.png'

img = cv2.imread(img_path)
mask = cv2.imread(mask_path)

print(img.shape)

print(mask.shape)


# mask 语义分割标注，与原图大小相同
np.unique(mask)


plt.figure(figsize=(10, 6))
plt.imshow(mask*50)
plt.axis('off')
plt.show()

