import cv2

import cv2kmeans
import cv2contour


def kmeans(imagepath, k=2):
    """
    Util:
        kmeans 像素聚类
    :param imagepath: 图像路径
    :param k: 聚类中心数目
    :return: 聚类后的图像
    """
    cv2kmeans.main(imagepath, k)


def contour(imagepath):
    """
    Util:
        图像轮廓识别
    :param imagepath: 图像路径
    :return: 标记了轮廓的图像
    """
    cv2contour.main(imagepath)