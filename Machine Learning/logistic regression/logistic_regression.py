import numpy as np
import matplotlib.pyplot as plt

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


# use gradient descent to solve the logistic regression
def logistic_regression(dataMat, labelMat, a, step):
    m = dataMat.shape[0]
    theta = np.random.random(dataMat.shape[1]) - 0.5

    for i in range(step):
        hx = 1 / (1 + np.exp(-np.dot(theta, dataMat.T)))
        theta = theta - a * 1 / m * np.dot(hx - labelMat, dataMat)
    return theta


#show the line between a and error
error_list = []
a_list = []
for ii in range(100):
    if ii == 0:
        a_list.append(0.0001)
    else:
        a_list.append(a_list[ii - 1] * 1.01)
    dataMat, labelMat = loadDataSet('G:\PythonProject\machinelearninginaction\Ch05\horseColicTraining.txt')
    ones = np.ones(dataMat.shape[0])
    dataMat = np.insert(dataMat, 0, ones, 1)
    theta = logistic_regression(dataMat, labelMat, a_list[ii], 1000)

    dataMat, labelMat = loadDataSet('G:\PythonProject\machinelearninginaction\Ch05\horseColicTest.txt')
    ones = np.ones(dataMat.shape[0])
    dataMat = np.insert(dataMat, 0, ones, 1)
    y = np.dot(theta, dataMat.T)
    for i in range(y.shape[0]):
        if y[i] < 0:
            y[i] = 0
        else:
            y[i] = 1

    errornum = 0
    for i in range(y.shape[0]):
        if y[i] != labelMat[i]:
            errornum = errornum + 1
    print(errornum / labelMat.shape[0])
    error_list.append(errornum / labelMat.shape[0])

plt.plot(a_list,error_list)
plt.show()
