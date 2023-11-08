import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import cv2

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