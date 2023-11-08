import cv2
import numpy as np

# 加载两张图片
img1 = cv2.imread("/home/niu/mmsegmentation/outputs/FCN_predict_2018_02_mosaic_L15-1439E-1134N_5759_3655_13_32.png")
img2 = cv2.imread("/home/niu/mmsegmentation/outputs/FCN_predict_2020_01_mosaic_L15-1439E-1134N_5759_3655_13_32.png")


# 计算差值
diff = cv2.absdiff(img1, img2)

# 将差值图像转换为灰度图像
gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

# 对灰度图像进行阈值化处理
_, threshold = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

# 寻找不同之处
contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 绘制不同之处的边界框（当不同之处的像素数量超过30时）
i=300
for contour in contours:
    if cv2.contourArea(contour) > i:  # 检查轮廓的像素数量是否大于30
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img1, (x, y), (x+w, y+h), (0, 0, 255), 2)

# 保存结果图像
cv2.imwrite('result_'+str(i)+'.png', img1)