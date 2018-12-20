import numpy as np
import cv2


def loadDataSet(arrimg):
    """
    读取图片数据(三维向量b,g,r) 到 feature(一维列表[r,g,b]) 数据集
    :param arrimg: jpg图片的numpy数组形式
    :return: 特征向量组feature
    """
    row = arrimg.shape[0]
    col = arrimg.shape[1]

    # print("数组行数:",row)
    # print("数组列数:",col)

    # 特征向量集合
    features = []

    print("正在读取图片信息，请稍等......")

    # (行,列):读取像素点rgb数据
    for i in range(0, row):
        for j in range(0, col):
            r = arrimg[i, j, 2]
            g = arrimg[i, j, 1]
            b = arrimg[i, j, 0]
            features.append([r, g, b])

    # 转换成numpy矩阵计算形式，浮点型f
    feature = np.array(features, 'f')

    # print(feature,"以上为像素点集合")
    # print("像素点集合的shape:", feature.shape)

    return feature


def distance(vecA, vecB):
    """
    计算两个向量的距离:三维向量-->一维距离
    :param vecA: 向量A
    :param vecB: 向量B
    :return: 浮点型数据
    """
    return np.sqrt(np.power(vecA[0] - vecB[0], 2) + np.power(vecA[1] - vecB[1], 2) + np.power(vecA[2] - vecB[2], 2))


def sel_init_cen(features):
    """
    随机选择两个初始点
    :param features: 图像的特征向量集,[[r,g,b],[r,g,b]...]
    :return: 初始聚类中心 cen1 cen2
    """
    # 选取随机数
    rand_1 = (int)(np.random.random() * (features.shape[0]))
    rand_2 = (int)(np.random.random() * (features.shape[0]))

    # 选取初始中心
    centor_1 = features[rand_1]
    centor_2 = features[rand_2]

    print("初始聚类中心为:", centor_1, "+", centor_2)

    return centor_1, centor_2


def kmeans(node, centor1, centor2, class1, class2):
    """
    kmeans方法迭代计算聚类中心
    :param node: 待判断数据[r,g,b]
    :param centor1: 当前聚类中心
    :param centor2: 当前聚类中心
    :param class1: 属于类1的数据集
    :param class2: 属于类2的数据集
    :return: 新的聚类中心 cen1 cen2
    """
    dist1 = distance(node, centor1)
    dist2 = distance(node, centor2)

    # 判断新数据和两个聚类中心的距离
    if dist1 < dist2:
        print(node, "添加到class1")
        class1.append(node)
        centor1 = np.mean(class1, axis=0)

    elif dist2 < dist1:
        class2.append(node)
        print(node, "添加到class2")
        centor2 = np.mean(class2, axis=0)

    else:
        class1.append(node)
        class2.append(node)
        print(node, "添加到c1和c2")
        centor1 = np.mean(class1, axis=0)
        centor2 = np.mean(class2, axis=0)

    print("mean-class1", centor1)
    print("mean-class2", centor2)

    return centor1, centor2


def window_get_centor(arrimg):
    """
    第1种方式：window-->fisher判别方法获得聚类中心
    1、获得数据集类1，类2
    2、计算均值向量 mean1 mean2
    3、计算类内离散度矩阵 sw_1 sw_2
    4、计算总类内离散度矩阵 sw=sw_1+sw_2
    5、获得sw的逆矩阵 sw.I
    6、计算w=

    :param arrimg: 图片数据集
    :return: 结果聚类中心 cen1 cen2
    """
    # 图片长宽
    length = arrimg.shape[1]
    width = arrimg.shape[0]
    # 类1 的窗口
    cl1_x_str = (int)(0.44 * length)
    cl1_x_end = (int)(0.55 * length)
    cl1_y_str = (int)(0.44 * width)
    cl1_y_end = (int)(0.55 * width)
    # 类2 的窗口
    cl2_x_str = (int)(0.77 * length)
    cl2_x_end = (int)(0.88 * length)
    cl2_y_str = (int)(0.44 * width)
    cl2_y_end = (int)(0.55 * width)

    class1 = []
    class2 = []

    # 获取类1
    for row1 in range(cl1_x_str, cl1_x_end):
        for col1 in range(cl1_y_str, cl1_y_end):
            class1.append(arrimg[col1][row1])
    # 获取类2
    for row2 in range(cl2_x_str, cl2_x_end):
        for col2 in range(cl2_y_str, cl2_y_end):
            class2.append(arrimg[col2][row2])

    feature1 = np.array(class1, 'f')
    feature2 = np.array(class2, 'f')
    cen_1 = np.mean(feature1, axis=0)
    cen_2 = np.mean(feature2, axis=0)

    print("window计算类1中心为：", cen_1)
    print("window计算类2中心为：", cen_2)

    return cen_1, cen_2


def kmeans_get_centor(feature, cen_1, cen_2):
    """
    第2种方式：kmeans方法获得聚类中心
    :param feature: 数据集
    :param cen_1: 初始聚类中心
    :param cen_2: 初始聚类中心
    :return: 结果聚类中心 cen1 cen2
    """
    # 建立两个聚类的空集合，注意：引入更正数据[0,0,0]，防止数据过大
    class1 = []
    class2 = []
    class1.append([0, 0, 0])
    class2.append([0, 0, 0])

    # 第一种方式：设置大步长，减少计算时间
    for i in range(0, feature.shape[0] - 1, 100):
        print("当前为第", i, "轮")
        # 使用kmeans计算方法
        centor1, centor2 = kmeans(feature[i], cen_1, cen_2, class1, class2)
        cen_1 = centor1
        cen_2 = centor2

        # 第二种方式：计算两次迭代之间的均方差---->当均方差<0.0001时，停止迭代
        pass

    return cen_1, cen_2


def image2k(centor1, centor2):
    """
    根据聚类中心进行图像二分类
    :param centor1: 结果聚类中心
    :param centor2: 结果聚类中心
    :return: 显示图像
    """
    img2 = cv2.imread("IMGP8080.JPG")
    print(img2.shape)
    row = img2.shape[0]
    col = img2.shape[1]

    # (行,列):设置像素点bgr数据
    for i in range(0, row):
        for j in range(0, col):
            print("图像分类已进行到: 第", i, "行+", j, "列")
            # 类1 黑色 类型[b,g,r]
            if distance(img2[i][j], centor1) < distance(img2[i][j], centor2):
                img2[i][j] = [0, 0, 0]
            # 类2 红色 类型[b,g,r]
            else:
                img2[i][j] = [0, 0, 255]

    # 窗口,调整图像大小
    win = cv2.namedWindow('img win', flags=0)
    cv2.imshow('img win', img2)
    cv2.waitKey(0)


def sel_function(n, feature, arrimg, init_cen_1, init_cen_2):
    """
    三选一，获得聚类中心
    :param n: 哪种方式？1,2,3
    :param feature: 特征向量数据集
    :param arrimg: 图片数据集
    :param init_cen_1: 初始聚类中心
    :param init_cen_2: 初始聚类中心
    :return: 聚类中心 cen1 cen2
    """
    if n == 1:
        # 第1种方法 window获得聚类中心
        cen_1, cen_2 = window_get_centor(arrimg)
    elif n == 2:
        # 第2种方法 kmeans获得聚类中心
        cen_1, cen_2 = kmeans_get_centor(feature, init_cen_1, init_cen_2)
    elif n == 3:
        # 第3种方法直接获得计算结果，节省时间
        cen_1 = [95.81982617, 69.23093989, 57.99697231]
        cen_2 = [207.61375807, 152.36148107, 124.72904938]

    else:
        cen_1 = init_cen_1
        cen_2 = init_cen_2
        print("无此方法，请输入1,2,3")

    return cen_1, cen_2


def run():
    """
    主程序
    """
    img = cv2.imread("IMGP8080.JPG")
    arrimg = np.array(img)
    # 获取像素点的集合
    feature = loadDataSet(arrimg)
    # 获取初始聚类中心
    init_cen_1, init_cen_2 = sel_init_cen(feature)
    # 三种方式获得聚类中心
    n = input("请选择function:\n"
              "1-->第五题窗口法\n"
              "2-->第六题kmeans聚类法\n"
              "3-->快速验证(使用计算好的聚类中心)\n")
    cen1, cen2 = sel_function(n, feature, arrimg, init_cen_1, init_cen_2)
    # 打印结果
    print("最终结果:", cen1, cen2)
    # 图片分类及展示图片
    image2k(cen1, cen2)


if __name__ == '__main__':
    run()
