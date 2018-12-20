import sys
import numpy as np
import cv2


def loadDataSet(arrimg):
    """
    读取numpy.array()图像数据

        from: array([[b,g,r],[b,g,r]...],
                    ....................
                    [[b,g,r],[b,g,r]...])

        to: [[r,g,b],[r,g,b]...]

    :param arrimg: array([[b,g,r],[b,g,r]...],[[],[]],......)
    :return: features=[[r,g,b],[r,g,b]...]
    """
    print("正在读取图片信息，请稍等......")

    row = arrimg.shape[0]
    col = arrimg.shape[1]

    features = []

    # read [r,g,b]
    for i in range(0, row):
        for j in range(0, col):
            r = arrimg[i, j, 2]
            g = arrimg[i, j, 1]
            b = arrimg[i, j, 0]
            features.append([r, g, b])

    features = np.array(features, 'f')
    return features


def distance(vecA, vecB):
    """
    计算rgb向量的欧式距离

    :param vecA: valueof[r,g,b]
    :param vecB: valueof[r.g.b]
    :return: dist
    """
    return np.sqrt(np.power(vecA[0] - vecB[0], 2) + np.power(vecA[1] - vecB[1], 2) + np.power(vecA[2] - vecB[2], 2))


def sel_init_cen(features, k):
    """
    随机选择K个初始聚类中心

    :param features: [[r,g,b],[r,g,b]...]
    :return: centors=[cen1,cen2...]
    """
    # 选取随机数
    rands = [(int)(np.random.random() * (features.shape[0])) for _ in range(k)]
    # 选取初始中心
    centors = [features[rands[i]] for i in range(k)]
    return centors


def get_centor(feature, centors):
    """
    迭代计算聚类中心

    :param node: 待判断数据[r,g,b]
    :param centors: init[cen1,cen2...]
    :param classes: [[node of class1],[node of class2],......[node of classk]]
    :return: cens=[cen1,cen2...]
    """
    k = len(centors)

    # 建立k个类别数据的空集合
    classes = [[] for _ in range(k)]

    # 设置大步长，减少计算时间
    for i in range(0, feature.shape[0] - 1, 100):

        # node到k个聚类中心的距离
        dists = [distance(feature[i], centor) for centor in centors]

        # 判为距离最近的类别，并重新计算聚类中心(平均值)
        for j in range(k):
            if min(dists) == distance(feature[i], centors[j]):
                classes[j].append(feature[i])
                centors[j] = np.mean(classes[j], axis=0)
                break

    return centors


def image2k(imagepath, centors):
    """
    根据聚类中心进行图像分类

    :param centors: 聚类中心
    :return: 显示图像
    """
    img2 = cv2.imread(imagepath)
    row = img2.shape[0]
    col = img2.shape[1]
    k = len(centors)

    print(centors)
    # 按灰度值从小到大排序
    centors=sorted([centor.tolist() for centor in centors])

    # 定义颜色库 8
    colors = [[0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255],
              [255, 255, 0], [0, 255, 255], [255, 0, 255]]

    # (行,列):根据类别设置像素bgr数据
    for i in range(0, row):
        for j in range(0, col):
            print("图像分类已进行到:", i+1, "/", row, "行", j+1, "/", col, "列")
            # 当前像素到k个聚类中心的距离
            dists = [distance(img2[i][j], centor) for centor in centors]
            for ks in range(k):
                if min(dists) == distance(img2[i][j], centors[ks]):
                    img2[i][j] = colors[ks % len(colors)]

    # 窗口,调整图像大小
    win = cv2.namedWindow('kmeans', flags=0)
    cv2.imshow('kmeans', img2)
    cv2.waitKey(0)


def main(imagepath, k):
    """
    程序入口
    """
    if k < 2 | k > 9:
        print('k is error')
        sys.exit(0)
    # numpy获取图像bgr数组
    arrimg = np.array(cv2.imread(imagepath))
    # 获取[[r,g,b]...]
    feature = loadDataSet(arrimg)
    # 获取k个随机初始聚类中心
    init_cens = sel_init_cen(feature, k)
    # 计算k个聚类中心
    cens = get_centor(feature, init_cens)
    # 显示k分类的图像
    image2k(imagepath, cens)