import cv2
import numpy as np

# 二值形态学运算
def morphology(img):
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (20,14)) # 腐蚀矩阵
    iFushi = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel1)  # 对文字腐蚀运算
    cv2.imshow('fushi', iFushi)

    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))  # 膨胀矩阵
    iPengzhang = cv2.morphologyEx(iFushi, cv2.MORPH_ERODE, kernel2)  # 对背景进行膨胀运算
    cv2.imshow('pengzhang', iPengzhang)

    # 背景图和二分图相减-->得到文字
    jian = np.abs(iPengzhang - img)
    cv2.imshow("jian", jian)

    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3 , 6))  # 膨胀
    iWenzi = cv2.morphologyEx(jian, cv2.MORPH_DILATE, kernel3)  # 对文字进行膨胀运算
    cv2.imshow('wenzi', iWenzi)

img = cv2.imread("TEST.tif")
# 1、消除椒盐噪声：
# 中值滤波器
median = cv2.medianBlur(img, 5)
# 消除噪声图
cv2.imshow("median-image", median)
# 转化为灰度图
Grayimg = cv2.cvtColor(median, cv2.COLOR_RGB2GRAY)
# 2、直方图均衡化：
hist = cv2.equalizeHist(Grayimg)
cv2.imshow('hist',hist)
# 3、二值化处理：
# 阈值为140
ret, binary = cv2.threshold(hist, 140, 255,cv2.THRESH_BINARY)
cv2.imshow("binary-image",binary)
# 二值形态处理
morphology(binary)

cv2.waitKey(0)
