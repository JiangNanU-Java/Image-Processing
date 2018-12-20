## 提取椒盐噪声下的车牌号

![image](https://img-blog.csdn.net/20180417231115765?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dzaDU5NjgyMzkxOQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

python程序：
```Python
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
```
python程序步骤：
（一）读入图像数据    
![image](https://img-blog.csdn.net/20180417231109349?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dzaDU5NjgyMzkxOQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)  
使用OpenCV库读入tiff图像
```python
img =cv2.imread("TEST.tif")
```
 

（二）消除椒盐噪声   
![image](https://img-blog.csdn.net/20180417231048796?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dzaDU5NjgyMzkxOQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
使用5*5的中值滤波器滤除椒盐噪声
```python
    median = cv2.medianBlur(img, 5)
```
 

（三）将图片数据类型转换为灰度图
```python
    Grayimg = cv2.cvtColor(median, cv2.COLOR_RGB2GRAY)
```
 

（四）对图像进行直方图均衡化处理   
![image](https://img-blog.csdn.net/20180417231141779?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dzaDU5NjgyMzkxOQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
```python
    hist = cv2.equalizeHist(Grayimg)
```


 

（五）对直方图均衡化后进行二值化处理   
![image](https://img-blog.csdn.net/20180417231155103?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dzaDU5NjgyMzkxOQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
选取阈值为140，大于140灰度值的像素置位255，低于140灰度值为0
```python
    ret, binary = cv2.threshold(hist, 140, 255,cv2.THRESH_BINARY)
```


 

（六）对二值化图像进行腐蚀：

腐蚀掉文字部分得到背景部分，然后再对背景进行膨胀

使用（20,14）的矩形进行腐蚀操作：
```python
kernel1= cv2.getStructuringElement(cv2.MORPH_RECT, (20,14))    iFushi = cv2.morphologyEx(img,cv2.MORPH_DILATE, kernel1)  
```
      

![image](https://img-blog.csdn.net/20180417231208939?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dzaDU5NjgyMzkxOQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![image](https://img-blog.csdn.net/20180417231216228?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dzaDU5NjgyMzkxOQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
 

（七）将二分图像与腐蚀后的背景图相减

得到有效信息部分与部分噪声
```python
    jian = np.abs(iPengzhang - img)
```
  
![image](https://img-blog.csdn.net/20180417231227305?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dzaDU5NjgyMzkxOQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
 

（八）对文字部分进行膨胀，得到较清晰的图像
```

    kernel3 =cv2.getStructuringElement(cv2.MORPH_RECT, (3 , 6))

    iWenzi =cv2.morphologyEx(jian, cv2.MORPH_DILATE, kernel3)
    
```
  
![image](https://img-blog.csdn.net/20180417231236441?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dzaDU5NjgyMzkxOQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
 

（1）滤除椒盐噪声可以使用均值滤波器或中值滤波器:

分析可知：均值滤波器会引入噪声的影响，而中值滤波器可以有效滤除噪声的影响。

试验了两种滤波器后，发现均值滤波器相较中值滤波器较模糊，所以滤除椒盐噪声使用中值滤波器会好一点

（2）选取合适的阈值进行二值化处理，要做到使信息清晰，并且能够有效地分割边界。

（3）选取合适的运算矩阵进行二值形态学的运算，通过腐蚀掉有效区域，然后进行图像减法的操作，可以得到被腐蚀的区域。

（4）对于断裂处，可采用长方形的矩阵进行膨胀处理，对于黏连处，同样可以采用长方形进行腐蚀处理
