import numpy as np
import matplotlib.pyplot as plt
import torch


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


dataMat, labelMat = loadDataSet('G:\PythonProject\machinelearninginaction\Ch08\ex1.txt')
dataMat = dataMat[:, 1]

dataMat = torch.tensor(dataMat, dtype=torch.float) - 0.5
labelMat = torch.tensor(labelMat, dtype=torch.float) - 0.5
dataMat = dataMat.view([200, 1])
labelMat = labelMat.view([200, 1])

w = torch.randn(1, 1, dtype=torch.float, requires_grad=True)
b = torch.randn(1, 1, dtype=torch.float, requires_grad=True)

for i in range(1000):
    y_pred = dataMat.mm(w) + b
    loss = (y_pred - labelMat).pow(2).sum()

    loss.backward()

    with torch.no_grad():
        w -= (0.0001 * w.grad)
        b -= (0.0001 * b.grad)

        w.grad.zero_()
        b.grad.zero_()

    print(loss)

y = y_pred.detach().numpy()

plt.scatter(dataMat.numpy(), labelMat.numpy())
plt.scatter(dataMat.numpy(), y)
plt.show()
