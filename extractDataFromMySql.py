# _*_ coding:utf-8 _*_

# 导入模块
import pymysql
import analysisHtmlByBS4
from itertools import combinations

'''
为什么DVD机连在电视上会自动出仓？
---cvtenlp---dvd老是出仓怎么回事?
---cvtenlp---DVD机进出仓失灵是什么原因
---cvtenlp---把DVD光盘放进DVD中为什么关仓了又自动出仓呢?
---cvtenlp---dvd机为什么进仓后会自动出仓？
---cvtenlp---先科dvd影碟机老是自动进出仓,这是为什么
---cvtenlp---我的DVD影碟机进仓之后一会自动出仓是什么原因？
---cvtenlp---VCD放入碟片按进仓后，自动就出仓了。再按进仓又自动出仓。一...
---cvtenlp---DVD机按了进仓之后老是自动弹出

'''


def getCombinations(sentences,num):
    '''
    获取所有句子的num 组合
    :param sentences:
    :param num:
    :return:
    '''
    if len(sentences) <= 1:
        return None
    return list(combinations(sentences,num))


def extractDataFromMySql():
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
    num_fail=0
    fw = open("src_tgt.txt",mode="w",encoding="utf-8")
    #先写入columns_name
    while True:
        row = cursor.fetchone()
        if not row:
            break
        try:
            fq_1 = row[0].strip().replace("\n","").replace("\t","").replace(" ","").replace("\r\n","")
            fq_2 = row[1].strip().replace("\n","").replace("\t","").replace(" ","").replace("\r\n","")
            first_href_raw_text = row[2]
            #写入多行用writerows
            sen_list = [fq_1,fq_2]
            sen_list.extend(analysisHtmlByBS4.analysisHtml(first_href_raw_text))
            sen_set = set(sen_list)
            # 获取所有句子的2组合数
            combinations = getCombinations(sen_set, 2)
            if combinations is not None:
                for sen1,sen2 in getCombinations(sen_set,2):
                    fw.write("---xhm---".join([sen1,sen2])+"\n")
            num += 1
            if num%10000==0:
                print("请耐心等待，数据正在提取处理中===========",num)
        except Exception:
            print("警告：出现了处理错误。采取放弃该条记录数据")
            num_fail+=1


    print("共：",num,"行")


if __name__=="__main__":
    extractDataFromMySql()
