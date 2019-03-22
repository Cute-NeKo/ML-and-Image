import numpy as np
import os
import jieba
import sys
from sklearn.cluster import KMeans
import pymysql

conn = pymysql.connect(host='localhost', port=3306, user='root', passwd='55555', db='companysystem', charset='utf8')
cur = conn.cursor()

titles = []
max_word_array = []


def read_stop_wprds(file_name):  # 读取停用词
    with open(file_name, "r") as fp:
        words = [line.replace('\n', '') for line in fp.readlines()]
    return words


def read_texts():
    texts = []
    sql = 'SELECT * FROM intelligence '
    cur.execute(sql)
    for r in cur:
        texts.append(str(r[2]).replace(' ', ''))
        titles.append(str(r[1]))
    return texts


def del_stop_words(stop_words, text):  # 将停用词从文章中删除
    text_split = jieba.cut(text)
    new_text = []

    for t in text_split:
        if t not in stop_words:
            new_text.append(t)
    return new_text


def get_all_vector():  # 构建词袋空间
    stop_words = read_stop_wprds(r"G:\PythonProject\InformationSystem\text\stopkey.txt")

    texts = read_texts()

    texts_split = []
    for text in texts:
        texts_split.append(del_stop_words(stop_words, text.replace('\n', '')))

    docs = []
    word_set = set()
    for text in texts_split:
        docs.append(text)
        word_set |= set(text)
    word_list = list(word_set)

    docs_vsm = []
    for doc in docs:
        max_count = -1
        temp_vector = []
        for word in word_list:
            count = doc.count(word)
            if count > max_count:
                max_count = count
                max_word = word
            temp_vector.append(count * 1.0)
        max_word_array.append(max_word)
        docs_vsm.append(temp_vector)
    docs_array = np.array(docs_vsm)
    return docs_array


def get_tf_idf():
    docs_array = get_all_vector()
    column_sum = [float(len(np.nonzero(docs_array[:, i])[0])) for i in range(docs_array.shape[1])]
    column_sum = np.array(column_sum)
    column_sum = docs_array.shape[0] / column_sum
    idf = np.log(column_sum)
    idf = np.diag(idf)

    tf = []
    for doc_v in docs_array:
        if doc_v.sum() == 0:
            doc_v = doc_v / 1
        else:
            doc_v = doc_v / (doc_v.sum())
        tf.append(doc_v)
    tf = np.array(tf)
    tf_idf = np.dot(tf, idf)
    return tf_idf

#相似度函数
def gen_sim(A, B):
    num = float(np.dot(A, B.T))
    denum = np.linalg.norm(A) * np.linalg.norm(B)
    if denum == 0:
        denum = 1
    cosn = num / denum
    sim = 0.5 + 0.5 * cosn
    return sim


#求初始聚类中心
def randCent(dataSet, k):
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k, n)))  # create centroid mat
    for j in range(n):  # create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = np.mat(minJ + rangeJ * np.random.rand(k, 1))
    return centroids


def kMeans(dataSet, k, distMeas=gen_sim, createCent=randCent):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m, 2)))  # create mat to assign data points
    # to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)
    clusterChanged = True
    counter = 0
    while counter <= 50:
        counter += 1
        clusterChanged = False
        for i in range(m):  # for each data point assign it to the closest centroid
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        # print centroids
        for cent in range(k):  # recalculate centroids
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]  # get all the point in this cluster
            centroids[cent, :] = np.mean(ptsInClust, axis=0)  # assign centroid to mean
    return centroids, clusterAssment


def start(num):
    tf_idf = get_tf_idf()

    ###########################################################################
    # 程序用的代码
    # y_pred = KMeans(n_clusters=int(num)).fit_predict(tf_idf)
    # sql = 'UPDATE intelligence SET category=%(y)s WHERE title=%(title)s'
    # for i in range(y_pred.shape[0]):
    #     value = {'y': str(y_pred[i]), 'title': titles[i]}
    #     cur.execute(sql, value)
    # conn.commit()
    #####################################################################

    # for word in max_word_array:
    #     print(word)

    # sql = 'UPDATE intelligence SET category=%(y)s WHERE title=%(title)s'
    myCentroids, clustAssing = kMeans(tf_idf, num, gen_sim, randCent)
    for label, name in zip(clustAssing[:, 0], titles):
        print(int(label[0, 0]), name)

    # y_pred = KMeans(n_clusters=num).fit_predict(tf_idf)
    # for i in range(y_pred.shape[0]):
    #     print(str(y_pred[i]) + '  ' + titles[i])


start(4)
