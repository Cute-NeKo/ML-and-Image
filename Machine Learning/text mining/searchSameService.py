import jieba
import pymysql
import numpy as np
import sys

conn = pymysql.connect(host='localhost', port=3306, user='root', passwd='55555', db='companysystem', charset='utf8')
cur = conn.cursor()

titles = []


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


def get_sim(A, B):
    A = np.array(A)
    B = np.array(B)
    num = float(np.dot(A, B.T))
    denum = np.sqrt(np.sum(A ** 2)) * np.sqrt(np.sum(B ** 2))
    sim = num / denum
    # denum = np.linalg.norm(A) * np.linalg.norm(B)
    # if denum == 0:
    #     denum = 1
    # cosn = num / denum
    # sim = 0.5 + 0.5 * cosn
    return sim


def get_all_vector():  # 构建词袋空间
    stop_words = read_stop_wprds(r"G:\PythonProject\InformationSystem\text\stopkey.txt")

    texts = get_company()

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
        temp_vector = []
        for word in word_list:
            count = doc.count(word)
            temp_vector.append(count * 1.0)
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


def get_company():
    company_map = {}
    sql = 'SELECT * FROM company'

    cur.execute(sql)

    companys = []
    for c in cur:
        titles.append(str(c[1]))
        companys.append(str(c[3]))
    return companys


def start(this_company_name):
    tf_idf = get_tf_idf()
    k = 0
    for title in titles:
        if title == this_company_name:
            break
        k += 1
    similar = []
    company_map = {}
    for i in range(len(titles)):
        similar.append(get_sim(tf_idf[k], tf_idf[i]))
    for i in range(len(titles)):
        if i != k:
            company_map[titles[i]] = similar[i]

    company_map = sorted(company_map.items(), key=lambda d: d[1], reverse=True)
    for company in company_map:
        print(str(company[0]) + '$' + str(company[1]))


start(sys.argv[1])
# start('湖北省神龙地质工程勘察院')
