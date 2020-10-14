import psycopg2


def insert(algo, data_type, epoch, optimizer, dropout, logLoss, auc):
    # 创建连接对象
    conn = psycopg2.connect(database="aimp", user="postgres", password="postgres", host="10.106.130.132", port="30083")
    cur = conn.cursor()  # 创建指针对象
    # # 创建表
    # cur.execute("CREATE TABLE student(id integer,name varchar,sex varchar);")

    # 插入数据
    cur.execute(
        "INSERT INTO aimp_deep_ctr(algo,data_type,epochs,optimizer,dropout,log_loss,auc,insert_time)VALUES(%s,%s,%s,%s,%s,%s,%s,current_timestamp)",
        (algo, data_type, epoch, optimizer, dropout, logLoss, auc))

    # # 获取结果
    # cur.execute('SELECT * FROM student')
    # results = cur.fetchall()
    # print(results)

    # 关闭连接
    conn.commit()
    cur.close()
    conn.close()
