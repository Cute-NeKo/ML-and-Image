import numpy as np


# load data
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1  # get number of fields
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    dataMat = np.array(dataMat)
    labelMat = np.array(labelMat)
    return dataMat, labelMat


def sigmoid(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)

    return 1 / (1 + np.exp(-x))


# parameter: the hiddenlayer parameter is a list contain the hiddenlayer's num
def BPNN(dataMat, labelMat, hiddenlayer, a=0.01, inputlayre_num=1, step=10000):
    hiddenlayer = np.array(hiddenlayer)
    # the Initialization of weights
    theta = []
    for i in range(hiddenlayer.shape[0] + 1):
        if i == 0:
            theta1 = 2 * np.random.random((hiddenlayer[i] + 1, dataMat.shape[1])) - 1
        elif i == hiddenlayer.shape[0]:
            theta1 = 2 * np.random.random((inputlayre_num, hiddenlayer[i - 1] + 1)) - 1
        else:
            theta1 = 2 * np.random.random((hiddenlayer[i] + 1, hiddenlayer[i - 1] + 1)) - 1
        theta.append(theta1)
        # print(theta1.shape)

    # theta = np.array(theta)

    # start
    for i in range(step):
        a = [dataMat.T]
        for j in range(hiddenlayer.shape[0] + 1):
            aa = sigmoid(np.dot(theta[j], a[j]))
            # if j != hiddenlayer.shape[0]:
            #     one = np.ones(aa.shape[1])
            #     aa = np.insert(aa, 0, ones, 0)
            a.append(aa)
            # print(aa.shape)

        error = []
        delta = []
        error.insert(0, labelMat - a[len(a) - 1])
        delta.insert(0, error[0] * sigmoid(a[len(a) - 1], deriv=True))

        for j in range(hiddenlayer.shape[0]):
            # 得把加上的那一列去掉
            # if j != 0:
            #     delta2 = np.delete(delta[0], 0, 0)
            # else:
            #     delta2 = delta[0]
            error1 = np.dot(theta[len(theta) - 1 - j].T, delta[0])
            delta1 = error1 * sigmoid(a[len(a) - 2 - j], deriv=True)
            delta.insert(0, delta1)
            # print(delta1.shape)

        # update weights
        for j in range(len(theta)):
            theta[j] += np.dot(delta[j], a[j].T)

    return theta


def predict(theta, x):
    a = x.T
    for l in range(0, len(theta)):
        a = sigmoid(np.dot(theta[l], a))
    return a


dataMat, labelMat = loadDataSet('G:\PythonProject\machinelearninginaction\Ch05\horseColicTraining.txt')
ones = np.ones(dataMat.shape[0])
dataMat = np.insert(dataMat, 0, ones, 1)
theta = BPNN(dataMat, labelMat, [4, 4, 4], 0.1, 1, 10000)

x, y = loadDataSet('G:\PythonProject\machinelearninginaction\Ch05\horseColicTest.txt')
ones = np.ones(x.shape[0])
x = np.insert(x, 0, ones, 1)
result = predict(theta, x)
print(result)
