import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from mmseg.apis import init_model, inference_model, show_result_pyplot
import mmcv
import cv2


#载入模型配置文件
config_file = './Zihao-Configs/WHUDataset_UNet_20230818.py'

# 模型 checkpoint 权重文件
checkpoint_file = './work_dirs/WHUDataset-UNet/iter_40000.pth'
#提取epoch数
import re


pattern = r'(\d+)'
match = re.search(pattern, checkpoint_file)
if match:
    epoch = match.group(1)
    print(epoch)
else:
    print("No number found in the file path.")

# device = 'cpu'
device = 'cuda:0'

model = init_model(config_file, checkpoint_file, device=device)

img_path = './global_monthly_2017_07_mosaic_L15-1669E-1160N_6678_3548_13_50.tif'

img_bgr = cv2.imread(img_path)

plt.figure(figsize=(8, 8))
plt.imshow(img_bgr[:,:,::-1])
plt.show()

result = inference_model(model, img_bgr)
result.keys()

pred_mask = result.pred_sem_seg.data[0].cpu().numpy()
pred_mask.shape

np.unique(pred_mask)


plt.figure(figsize=(8, 8))
plt.imshow(pred_mask)

save_path='./outputs/spaceNet50_'+epoch+'_predict.png'


plt.figure(figsize=(8, 8))
plt.imshow(pred_mask)
plt.savefig(save_path)
plt.show()