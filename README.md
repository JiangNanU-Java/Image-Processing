# Image Processing 基于人工智能算法的图像处理程序集合

## ImagePigmentation 图像色素分割

根据输入的k值进行Kmeans聚类算法实现对图像的像素分类

* kmeans算法使用基本数据结构实现,未调用kmeans算法库

## LicensePlateRecognition 车牌识别

![image](https://img-blog.csdn.net/20180417231115765?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dzaDU5NjgyMzkxOQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

1、选用合适的图像增强方法对以下给定图像进行增强操作以获取清晰图像；

2、对增强后的图像进行阈值处理，获得二值图像；

3、对二值图像进行形态学分析，提取有用信息区域（即只剩下字母和数字区域）

* 中值滤波消噪
* 直方图均衡化
* 图像二值化
* 二值形态学运算
* 图像开闭运算

## ImageTargetExtraction 图像目标提取

静态图像目标提取 ：识别有缺陷的药片

## DynamicTargetExtraction 视频动态目标提取 

视频动态目标提取 ：识别行人和车辆