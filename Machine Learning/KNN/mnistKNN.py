import numpy as np
import struct
import sys
from collections import namedtuple
import cv2


class KdNode(object):
    def __init__(self, dom_elt, split, left, right):
        self.dom_elt = dom_elt  # k维向量节点(k维空间中的一个样本点)
        self.split = split  # 整数（进行分割维度的序号）
        self.left = left  # 该结点分割超平面左子空间构成的kd-tree
        self.right = right  # 该结点分割超平面右子空间构成的kd-tree


class KdTree(object):
    def __init__(self, data):
        k = data.shape[1]  # 数据维度

        def CreateNode(split, data_set):  # 按第split维划分数据集exset创建KdNode
            if data_set.shape[0] < 1:  # 数据集为空
                return None

            data_set = data_set[data_set[:, split].argsort()]
            split_pos = len(data_set) // 2  # //为Python中的整数除法
            median = data_set[split_pos]  # 中位数分割点
            split_next = (split + 1) % k  # cycle coordinates
            # 递归的创建kd树
            return KdNode(median, split,
                          CreateNode(split_next, data_set[:split_pos, :]),  # 创建左子树
                          CreateNode(split_next, data_set[split_pos + 1:, :]))  # 创建右子树

        self.root = CreateNode(0, data)  # 从第0维分量开始构建kd树,返回根节点


# KDTree的前序遍历
def preorder(root):
    print(root.dom_elt)
    if root.left:  # 节点不为空
        preorder(root.left)
    if root.right:
        preorder(root.right)


# 定义一个namedtuple,分别存放最近坐标点、最近距离和访问过的节点数
result = namedtuple("Result_tuple", "nearest_point  nearest_dist  nodes_visited")


def find_nearest(tree, point):
    k = point.shape[0]  # 数据维度

    def travel(kd_node, target, max_dist):
        if kd_node is None:
            return result([0] * k, float("inf"), 0)  # python中用float("inf")和float("-inf")表示正负无穷

        nodes_visited = 1

        s = kd_node.split  # 进行分割的维度
        pivot = kd_node.dom_elt  # 进行分割的“轴”

        if target[s] <= pivot[s]:  # 如果目标点第s维小于分割轴的对应值(目标离左子树更近)
            nearer_node = kd_node.left  # 下一个访问节点为左子树根节点
            further_node = kd_node.right  # 同时记录下右子树
        else:  # 目标离右子树更近
            nearer_node = kd_node.right  # 下一个访问节点为右子树根节点
            further_node = kd_node.left

        temp1 = travel(nearer_node, target, max_dist)  # 进行遍历找到包含目标点的区域

        nearest = temp1.nearest_point  # 以此叶结点作为“当前最近点”
        dist = temp1.nearest_dist  # 更新最近距离

        nodes_visited += temp1.nodes_visited

        if dist < max_dist:
            max_dist = dist  # 最近点将在以目标点为球心，max_dist为半径的超球体内

        temp_dist = abs(pivot[s] - target[s])  # 第s维上目标点与分割超平面的距离
        if max_dist < temp_dist:  # 判断超球体是否与超平面相交
            return result(nearest, dist, nodes_visited)  # 不相交则可以直接返回，不用继续判断

        # ----------------------------------------------------------------------
        # 计算目标点与分割点的欧氏距离
        temp_dist = np.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(pivot, target)))

        if temp_dist < dist:  # 如果“更近”
            nearest = pivot  # 更新最近点
            dist = temp_dist  # 更新最近距离
            max_dist = dist  # 更新超球体半径

        # 检查另一个子结点对应的区域是否有更近的点
        temp2 = travel(further_node, target, max_dist)

        nodes_visited += temp2.nodes_visited
        if temp2.nearest_dist < dist:  # 如果另一个子结点内存在更近距离
            nearest = temp2.nearest_point  # 更新最近点
            dist = temp2.nearest_dist  # 更新最近距离

        return result(nearest, dist, nodes_visited)

    return travel(tree.root, point, float("inf"))  # 从根节点开始递归


# 利用opencv获取图像hog特征
def get_hog_features(trainset):
    features = []

    hog = cv2.HOGDescriptor('xml/hog.xml')

    for img in trainset:
        img = np.reshape(img, (28, 28))
        cv_img = img.astype(np.uint8)

        hog_feature = hog.compute(cv_img)
        # hog_feature = np.transpose(hog_feature)
        features.append(hog_feature)

    features = np.array(features)
    features = np.reshape(features, (-1, 324))

    return features


def loadImageSet(filename):
    binfile = open(filename, 'rb')  # 读取二进制文件
    buffers = binfile.read()

    head = struct.unpack_from('>IIII', buffers, 0)  # 取前4个整数，返回一个元组

    offset = struct.calcsize('>IIII')  # 定位到data开始的位置
    imgNum = head[1]
    width = head[2]
    height = head[3]

    bits = imgNum * width * height  # data一共有60000*28*28个像素值
    bitsString = '>' + str(bits) + 'B'  # fmt格式：'>47040000B'

    imgs = struct.unpack_from(bitsString, buffers, offset)  # 取data数据，返回一个元组

    binfile.close()
    imgs = np.reshape(imgs, [imgNum, width * height])  # reshape为[60000,784]型数组

    return imgs, head


def loadLabelSet(filename):
    binfile = open(filename, 'rb')  # 读二进制文件
    buffers = binfile.read()

    head = struct.unpack_from('>II', buffers, 0)  # 取label文件前2个整形数

    labelNum = head[1]
    offset = struct.calcsize('>II')  # 定位到label数据开始的位置

    numString = '>' + str(labelNum) + "B"  # fmt格式：'>60000B'
    labels = struct.unpack_from(numString, buffers, offset)  # 取label数据

    binfile.close()
    labels = np.reshape(labels, [labelNum])  # 转型为列表(一维数组)

    return labels, head


def KNN(data, label, x, k):
    numSamples = data.shape[0]

    ## step 1: calculate Euclidean distance
    diff = np.tile(x, (numSamples, 1)) - data
    squareDiff = diff ** 2
    squareDist = np.sum(squareDiff, axis=1)
    distance = squareDist ** 0.5

    ## step 2: sort the distance
    sortedDistIndices = np.argsort(distance)

    classCount = {}
    for i in range(k):
        ## step 3: choose the min k distance
        voteLabel = label[sortedDistIndices[i]]

        ## step 4: count the times labels occur
        # when the key voteLabel is not in dictionary classCount, get()
        # will return 0
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key
    return maxIndex


if __name__ == '__main__':
    file1 = 'E:/数据集/mnist/train-images.idx3-ubyte'
    file2 = 'E:/数据集/mnist/train-labels.idx1-ubyte'
    file3 = 'E:/数据集/mnist/t10k-images.idx3-ubyte'
    file4 = 'E:/数据集/mnist/t10k-labels.idx1-ubyte'

    data, data_head = loadImageSet(file1)
    label, label_head = loadLabelSet(file2)
    x, x_head = loadImageSet(file3)
    y, y_head = loadLabelSet(file4)

    data=get_hog_features(data)
    x=get_hog_features(x)

    data[data > 0] = 1
    data[data == 0] = 0
    x[x > 0] = 1
    x[x == 0] = 0
    for i in range(x.shape[0]):
        precient = KNN(data, label, x[i, :], 10)

        print(y[i], ':', precient)




    # data = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
    # data = np.array(data)
    # kd = KdTree(data)
    # preorder(kd.root)
    # ret2 = find_nearest(kd, np.array([5, 5]))
    # print(ret2)
