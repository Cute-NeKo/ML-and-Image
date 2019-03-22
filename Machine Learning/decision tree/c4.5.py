from sklearn import datasets
import numpy as np


class iris:
    def __init__(self):
        self.data = datasets.load_iris()
        self.x = np.mat(self.data.data)
        self.y = np.mat(self.data.target)
        self.train = np.concatenate((self.x, self.y.T), axis=1)

    def getShannon(self, train):
        counts = {}
        sum = len(train.tolist())

        for i in self.train.tolist():
            if i[-1] not in counts.keys():
                counts[i[-1]] = 0
            counts[i[-1]] += 1

        shannon = 0
        for i in counts.keys():
            shannon -= counts[i] / sum * np.log(counts[i] / sum)
        return shannon

    def splitDataSet(self, train, axis):
        leftList = []
        rightList = []
        mid = train.mean(0)[0, axis]
        for i in train.tolist():
            if i[axis] < mid:
                list2 = i[:axis]
                list2.extend(i[axis + 1:])
                leftList.append(list2)
            else:
                list2 = i[:axis]
                list2.extend(i[axis + 1:])
                rightList.append(list2)
        return np.mat(leftList), np.mat(rightList)

    def chooseBestFeature1(self, train):  # ID3
        numFeatures = train[0:1, :].shape[1] - 1
        baseEntropy = self.getShannon(train)
        bestFeature = -1
        bestInfoGain = 0.0
        for i in range(numFeatures):
            [leftData, rightData] = self.splitDataSet(train, i)
            newEntropy = 0.0
            newEntropy += self.getShannon(leftData) * leftData.shape[0] / train.shape[0]
            newEntropy += self.getShannon(rightData) * rightData.shape[0] / train.shape[0]
            infoGain = baseEntropy - newEntropy
            if infoGain > bestInfoGain:
                bestInfoGain = infoGain
                bestFeature = i
        return bestFeature

    def majorityCnt(self, classList):
        classCount = {}
        for i in classList:
            if i not in classCount.keys():
                classCount[i] = 0
            classCount[i] += 1
        max = 0
        t = -1
        for i in classCount.keys():
            if classCount[i] > max:
                max = classCount[i]
                t = i
        return t

    def createTree(self, train, labels):
        classList = [i.tolist()[0][0] for i in train[:, -1]]
        if classList.count(classList[0]) == len(classList):
            return classList[0]
        if train.shape[1] == 1:
            return self.majorityCnt(classList)
        bestFeature = self.chooseBestFeature1(train)
        bestFeatureLabel = labels[bestFeature]
        myTree = {bestFeatureLabel: {}}
        del labels[bestFeature]
        [leftData, rightData] = self.splitDataSet(train, bestFeature)
        subLabels1 = labels[:]
        subLabels2 = labels[:]
        myTree[bestFeatureLabel]['<' + str(train.mean(0)[0, bestFeature])] = self.createTree(leftData, subLabels1)
        myTree[bestFeatureLabel]['>' + str(train.mean(0)[0, bestFeature])] = self.createTree(rightData, subLabels2)
        return myTree

    def show(self):
        print(self.createTree(self.train, ['SepalL', 'SepalW', 'PetalL', 'PetalW']))

    def classify(self, myTree):
        myTree = self.createTree(self.train, ['SepalL', 'SepalW', 'PetalL', 'PetalW'])


a = iris()
a.show()
