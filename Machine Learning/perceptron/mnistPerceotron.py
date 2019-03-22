import cv2
import numpy as np
import struct
import random
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

study_step = 0.001  # 学习步长
study_total = 10000  # 学习次数
feature_length = 324  # hog特征维度
object_num = 2  # 分类的数字


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


def Train(trainset, train_labels):
    # 获取参数
    trainset_size = train_labels.shape[0]

    # 初始化 w,b
    w = np.zeros((feature_length, 1))
    b = 0

    study_count = 0  # 学习次数记录，只有当分类错误时才会增加
    nochange_count = 0  # 统计连续分类正确数，当分类错误时归为0
    nochange_upper_limit = 100000  # 连续分类正确上界，当连续分类超过上界时，认为已训练好，退出训练

    while True:
        nochange_count += 1
        if nochange_count > nochange_upper_limit:
            break

        # 随机选的数据
        index = random.randint(0, trainset_size - 1)
        img = trainset[index]
        label = train_labels[index]

        # 计算yi(w*xi+b)
        yi = int(label != object_num) * 2 - 1  # 如果等于object_num, yi= 1, 否则yi=-1
        result = yi * (np.dot(img, w) + b)

        # 如果yi(w*xi+b) <= 0 则更新 w 与 b 的值
        if result <= 0:
            img = np.reshape(trainset[index], (feature_length, 1))  # 为了维数统一，需重新设定一下维度

            w += img * yi * study_step  # 按算法步骤3更新参数
            b += yi * study_step

            study_count += 1
            if study_count > study_total:
                break
            nochange_count = 0

    return w, b


def Predict(testset, w, b):
    predict = []
    for img in testset:
        result = np.dot(img, w) + b
        result = result > 0

        predict.append(result)

    return np.array(predict)


if __name__ == '__main__':
    time_1 = time.time()
    file1 = 'E:/数据集/mnist/train-images.idx3-ubyte'
    file2 = 'E:/数据集/mnist/train-labels.idx1-ubyte'
    file3 = 'E:/数据集/mnist/t10k-images.idx3-ubyte'
    file4 = 'E:/数据集/mnist/t10k-labels.idx1-ubyte'

    data, data_head = loadImageSet(file1)
    label, label_head = loadLabelSet(file2)
    x, x_head = loadImageSet(file3)
    y, y_head = loadLabelSet(file4)

    features = get_hog_features(data)
    x = get_hog_features(x)

    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(features, label, test_size=0.33,

                                                                                random_state=23323)

    time_2 = time.time()
    print('read data cost ', time_2 - time_1, ' second', '\n')

    print('Start training')
    w, b = Train(train_features, train_labels)
    time_3 = time.time()
    print('training cost ', time_3 - time_2, ' second', '\n')

    print('Start predicting')
    test_predict = Predict(test_features, w, b)
    time_4 = time.time()
    print('predicting cost ', time_4 - time_3, ' second', '\n')

    for i in range(test_labels.shape[0]):
        if test_labels[i] == object_num:
            test_labels[i] = 0
        else:
            test_labels[i] = 1

    score = accuracy_score(test_labels, test_predict)
    print("The accruacy socre is ", score)

    #########################################################
    print('Start predicting')
    test_predict = Predict(x, w, b)

    for i in range(y.shape[0]):
        if y[i] == object_num:
            y[i] = 0
        else:
            y[i] = 1

    score = accuracy_score(y, test_predict)
    print("The accruacy socre is ", score)

