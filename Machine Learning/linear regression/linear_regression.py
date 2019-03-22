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


# the standard equation of the regression
def standard_equation(dataMat, labelMat):
    theta = np.dot(np.dot(np.linalg.inv(np.dot(dataMat.T, dataMat)), dataMat.T), labelMat)
    y = np.dot(dataMat, theta)
    return y


# use gradient descent to solve the regression
def gradient_descent(dataMat, labelMat, a, step):
    m = dataMat.shape[0]
    theta = np.zeros(dataMat.shape[1])
    for i in range(step):
        theta = theta - a * 1 / m * np.dot(dataMat.T, (np.dot(dataMat, theta) - labelMat))

    y = np.dot(dataMat, theta)
    return y


dataMat, labelMat = loadDataSet('G:\PythonProject\machinelearninginaction\Ch08\ex1.txt')

plt.scatter(dataMat[:, 1], labelMat)

y = gradient_descent(dataMat, labelMat, 0.3, 100)
plt.plot(dataMat[:, 1], y, 'r')
plt.show()
