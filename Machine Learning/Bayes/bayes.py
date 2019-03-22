import numpy as np
import re
import random



class bayes:
    def loadDataSet(self):
        postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                       ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                       ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                       ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                       ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                       ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
        classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
        return postingList, classVec

    def createVocabList(self, dataSet):
        vocabSet = set([])
        for doc in dataSet:
            vocabSet = vocabSet | set(doc)
        return list(vocabSet)

    def setOfwords2Vec(self, vocabList, inputSet):
        returnVec = [0] * len(vocabList)
        for word in inputSet:
            if word in vocabList:
                returnVec[vocabList.index(word)] += 1  # 词袋模型+=1 词集模型=1
            else:
                print("the word: {} is not in my Vocabulary".format(word))
        return returnVec

    def trainNB0(self, trainMatrix, trainCategory):
        numTrainDocs = len(trainMatrix)
        numWords = len(trainMatrix[0])
        pAbusive = np.sum(trainCategory) / float(numTrainDocs)
        p0num = np.ones(numWords)
        p1num = np.ones(numWords)
        p0Denom = 2.0
        p1Denom = 2.0
        for i in range(numTrainDocs):
            if trainCategory[i] == 1:
                p1num += trainMatrix[i]
                p1Denom += np.sum(trainMatrix[i])
            else:
                p0num += trainMatrix[i]
                p0Denom += np.sum(trainMatrix[i])
        p1Vect = np.log(p1num / p1Denom)
        p0Vect = np.log(p0num / p0Denom)
        return p0Vect, p1Vect, pAbusive

    def classifyNB(self, vec2Classify, p0Vec, p1Vec, pClass1):
        p1 = np.sum(vec2Classify * p1Vec) + np.log(pClass1)
        p0 = np.sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
        if p1 > p0:
            return 1
        else:
            return 0

    def testingNB(self):
        listOPosts, listClasses = self.loadDataSet()
        myVocabList = self.createVocabList(listOPosts)
        trainMat = []
        for postinDoc in listOPosts:
            trainMat.append(self.setOfwords2Vec(myVocabList, postinDoc))
        p0V, p1V, pAb = self.trainNB0(np.array(trainMat), np.array(listClasses))
        testEntry = ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please']
        thisDoc = np.array(self.setOfwords2Vec(myVocabList, testEntry))
        print("classify is:" + str(self.classifyNB(thisDoc, p0V, p1V, pAb)))

    def textParse(self, bigString):
        listOfTkens = re.split(r'\w+', bigString)
        return [tok.lower() for tok in listOfTkens if len(tok) > 2]

    def spamTest(self):
        docList = []
        fullText = []
        classList = []
        for i in range(1, 26):
            wordList = self.textParse(open(r'G:\DATA\机器学习实战\Ch04\email\spam\{}.txt'.format(i)).read())
            docList.append(wordList)
            fullText.extend(wordList)
            classList.append(1)
            wordList = self.textParse(open(r'G:\DATA\机器学习实战\Ch04\email\ham\{}.txt'.format(i)).read())
            docList.append(wordList)
            fullText.extend(wordList)
            classList.append(0)
        vocabList = self.createVocabList(docList)
        trainingSet = [i for i in range(50)]
        testSet = []
        for i in range(10):
            randIndex = int(random.uniform(0, len(trainingSet)))
            testSet.append(trainingSet[randIndex])
            del (trainingSet[randIndex])
        trainMat = []
        trainClasses = []
        for docIndex in trainingSet:
            trainMat.append(self.setOfwords2Vec(vocabList, docList[docIndex]))
            trainClasses.append(classList[docIndex])
        p0V, p1V, pSpam = self.trainNB0(np.array(trainMat), np.array(trainClasses))
        errorCount = 0
        for docIndex in testSet:
            wordVector = self.setOfwords2Vec(vocabList, docList[docIndex])
            if self.classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
                errorCount += 1
        print('the error rate is:', float(errorCount) / len(testSet))
