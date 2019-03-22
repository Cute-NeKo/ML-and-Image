import numpy as np
import struct


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


def naive_bayes(data, label, x, y):
    # 求labels中每个label的先验概率
    P_y = {}
    for l in range(10):
        P_y[l] = np.sum(label == l) / float(label.shape[0])

    # 求label与feature同时发生的概率
    result = []
    ii = 0
    for xx in x:  # 对每一个输入处理
        P_xy = {}
        for yy in P_y.keys():
            y_index = (label == yy)
            data2 = data[y_index]
            g = 1
            for j in range(xx.shape[0]):
                num = np.sum(data2[:, j] == xx[j])
                g *= num / data2.shape[0]

            P_xy[yy] = g * P_y[yy]
        result.append(max(P_xy, key=P_xy.get))
        print(y[ii], ':', result[ii])
        ii += 1
    return np.array(result)


if __name__ == '__main__':
    file1 = 'E:/数据集/mnist/train-images.idx3-ubyte'
    file2 = 'E:/数据集/mnist/train-labels.idx1-ubyte'
    file3 = 'E:/数据集/mnist/t10k-images.idx3-ubyte'
    file4 = 'E:/数据集/mnist/t10k-labels.idx1-ubyte'

    data, data_head = loadImageSet(file1)
    label, label_head = loadLabelSet(file2)
    x, x_head = loadImageSet(file3)
    y, y_head = loadLabelSet(file4)

    print(data.shape)

    # 数据二值化
    data[data > 0] = 1
    data[data == 0] = 0
    x[x > 0] = 1
    x[x == 0] = 0

    result = naive_bayes(data, label, x, y)
