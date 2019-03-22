import pymysql

conn = pymysql.connect(host='localhost', port=3306, user='root', passwd='55555', db='companysystem', charset='utf8')
cur = conn.cursor()

sql = 'SELECT * FROM intelligence WHERE id=1'
cur.execute(sql)
for r in cur:
    print(str(r[2]))
