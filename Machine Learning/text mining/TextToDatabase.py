import os
import pymysql

conn = pymysql.connect(host='localhost', port=3306, user='root', passwd='55555', db='companysystem', charset='utf8')
cur = conn.cursor()


def read_texts():
    texts = []
    titles = []
    for parent, dirnames, filenames in os.walk("text\\train"):
        for filename in filenames:
            texts.append(open(parent + '\\' + filename).read())
            titles.append(filename.replace('.txt', ''))
    return titles, texts


def to_database():
    titles, texts = read_texts()

    sql = 'INSERT INTO intelligence (title, text) VALUES (%(title)s,%(text)s)'

    for i in range(len(titles)):
        value = {'title': titles[i], 'text': texts[i]}
        cur.execute(sql, value)
    conn.commit()


to_database()
