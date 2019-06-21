# _*_ coding:utf-8 _*_

# 导入模块
import pymysql

# 1.连接到mysql数据库
conn = pymysql.connect(host='172.17.168.47', user='root', password='root', db='baiduzhidao_wuzhihui', charset='utf8')
# localhost连接本地数据库 user 用户名 password 密码 db数据库名称 charset 数据库编码格式
# 2.创建游标对象
cursor = pymysql.cursors.SSCursor(conn)
# 3.组装sql语句 需要查询的MySQL语句
sql = "select first_question,second_question,first_href_raw_text from similarity_question_100000"
# 4.执行sql语句
cursor.execute(sql)


num=0


fw = open("test.txt",mode="w",encoding="utf-8")

#先写入columns_name



while True:
    row = cursor.fetchone()
    if not row:
        break
    fq_1=row[0].strip().replace("\n","").replace("\t","").replace(" ","").replace("\r\n","")
    fq_2=row[1].strip().replace("\n","").replace("\t","").replace(" ","").replace("\r\n","")
    first_href_raw_text=row[2]
    #写入多行用writerows
    sen_list=[fq_1,fq_2]
    sen_set=set(sen_list)
    fw.write("---cvtenlp---".join(sen_set)+"\n")
    num+=1
    if num%10000==0:
        print("请耐心等待，数据正在提取中===========",num)

print("共：",num,"行")