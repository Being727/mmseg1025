import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from mmseg.apis import init_model, inference_model, show_result_pyplot
import mmcv
import cv2


#载入模型配置文件
config_file = "/home/niu/mmsegmentation/whuconfigs/WHUDataset_FCN_2023-10-20.py"

# 模型 checkpoint 权重文件
checkpoint_file = "/home/niu/mmsegmentation/work_dirs/WHUDataset-FCN/iter_60000.pth"
#提取iters
import re

net_name="FCN"
pattern = r'(\d+)'
match = re.search(pattern, checkpoint_file)
if match:
    iters = match.group(1)
    print(iters)
else:
    print("No number found in the file path.")

# device = 'cpu'
device = 'cuda:0'

model = init_model(config_file, checkpoint_file, device=device)

img_path = "/home/niu/mmsegmentation/demo/global_monthly_2020_01_mosaic_L15-1439E-1134N_5759_3655_13_32.jpg"
start_index = img_path.find("monthly_")  # 找到"monthly_"的起始位置
end_index = img_path.find(".jpg")  # 找到".jpg"的位置

if start_index != -1 and end_index != -1:
    img_name = img_path[start_index + len("monthly_"):end_index]  # 提取"monthly"之后、".jpg"之前的字符
    print(img_name)
else:
    print("未找到'monthly_'或'.jpg'")
    

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
#plt.imshow(pred_mask)

save_path='./outputs/'+net_name+'_predict_'+img_name+'.png'
print(save_path)

plt.figure(figsize=(8, 8))
plt.imshow(pred_mask)
plt.savefig(save_path)
#plt.show()