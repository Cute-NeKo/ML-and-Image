import jieba
import numpy as np
import pymysql
import sys

conn = pymysql.connect(host='localhost', port=3306, user='root', passwd='55555', db='companysystem', charset='utf8')
cur = conn.cursor()

W = []  # 所有文本


def read_stop_wprds(file_name):  # 读取停用词
    with open(file_name, "r") as fp:
        words = [line.replace('\n', '') for line in fp.readlines()]
    return words


def del_stop_words(stop_words, text):  # 将停用词从文章中删除
    text_split = jieba.cut(text)
    new_text = []

    for t in text_split:
        if t not in stop_words:
            new_text.append(t)
    return new_text


def split_word(line):
    return list(jieba.cut(line))


def get_texts():
    texts = []
    titles = []
    sql = 'SELECT * FROM intelligence '
    cur.execute(sql)
    for r in cur:
        texts.append(str(r[2]))
        titles.append(str(r[1]))
    return titles, texts


def get_num_P(text_list, word):
    num = 0
    p_list = []
    i = 0
    for text in text_list:
        if word == text:
            num = num + 1
            p_list.append(i)
        i = i + 1
    return num, p_list


def get_distance(p_list):
    p_array = np.array(p_list)
    if p_array.size == 0:
        return 0
    elif p_array.size == 1:
        return 0  # 因为只有一个数的数组的标准差是0
    else:
        d = []
        d.append(1)
        for i in np.arange(p_array.size - 1):
            d.append((p_array[i + 1] - p_array[i]) / np.max(p_array))
        d = np.array(d)
        return np.sqrt(np.var(d)) / np.mean(d)


def get_TF(W, word):
    sum = 0
    k = 0
    for line in W:
        sum += len(line)
        for i in line:
            if i == word:
                k = k + 1
    TF = k / sum
    return TF


def get_IDF(word):  # 读取数据库每一个文章都要判断
    k = 0
    for w in W:
        for line in w:
            if word in line:
                k += 1
                break
    if k == 0:
        return 0
    else:
        return np.log((len(W) / k) + 0.01)


def get_similar(key_words, W):
    C = []
    D = []
    for texts in W:
        c = []
        d = []
        for key in key_words:
            num, p_list = get_num_P(texts, key)
            c.append(num)
            d.append(get_distance(p_list))
        C.append(c)
        D.append(d)

    x = 1  # 权重 不知道具体值
    C = np.array(C)
    D = np.array(x * D)
    T = np.ones(len(W))
    T[0] = 1.5
    TF = []
    IDF = []
    for key in key_words:
        TF.append(get_TF(W, key))
        IDF.append(get_IDF(key))
    TF = np.array(TF)
    IDF = np.array(IDF)

    TW = (TF * IDF) / np.sqrt(np.sum(TF ** 2) * (IDF ** 2))

    similar = np.dot(np.dot(T, (C - D)), TW.T)
    return 0.0 if np.isnan(similar) else similar


def start(sentence):
    stop_words = read_stop_wprds(r"G:\PythonProject\InformationSystem\text\stopkey.txt")
    key_words = del_stop_words(stop_words, sentence)

    titles, texts = get_texts()

    for text in texts:
        a = []
        lines = text.split('\n')
        for line in lines:
            a.append(split_word(line.strip()))
        W.append(a)

    similars = []
    for i in range(len(texts)):
        similars.append(get_similar(key_words, W[i]))
        # for i in range(len(texts)):
        #     print(str(titles[i]) + " " + str(similars[i]))

    sql = 'UPDATE intelligence SET similarity=%(similar)s WHERE title=%(title)s'
    for i in range(len(texts)):
        value = {'similar': str(similars[i]), 'title': titles[i]}
        cur.execute(sql, value)
    conn.commit()


start(sys.argv[1])
