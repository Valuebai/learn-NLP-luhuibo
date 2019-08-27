#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：LuckyHuibo
@Date   ：2019/8/20 20:03
@Desc   ：连接数据库，读取
=================================================='''
import pymysql
import re
import jieba


def cut(string):
    return ' '.join(jieba.cut(string))


def token(string):
    string = re.findall(
        '[\d|\w|\u3002 |\uff1f |\uff01 |\uff0c |\u3001 |\uff1b |\uff1a |\u201c |\u201d |\u2018 |\u2019 |\uff08 |\uff09 |\u300a |\u300b |\u3008 |\u3009 |\u3010 |\u3011 |\u300e |\u300f |\u300c |\u300d |\ufe43 |\ufe44 |\u3014 |\u3015 |\u2026 |\u2014 |\uff5e |\ufe4f |\uffe5]+',
        string)
    return ' '.join(string)


def deal(string):
    string = token(string)
    return cut(string)


# 从数据库中得到新闻语料库
def get_news_from_sql(host, user, password, database, port):
    print('开始连接数据库...')
    db = pymysql.connect(host, user, password, database, port, charset='utf8')  # 不添加charset，读取到的数据是乱码
    print('连接成功...')

    cursor = db.cursor()
    with open('news-sentences-xut.txt', 'r+', encoding='utf-8') as f:
        sql = """SELECT content from news_chinese"""
        try:
            cursor.execute(sql)
        except:
            # 如果发生异常，则回滚
            print("发生异常", Exception)
            db.rollback()
            return

        news = cursor.fetchall()
        print(news)

        for j in range(len(news)):
            data = news[j][0]
            text = deal(data)  # 处理中文的符号，此处处理不行，需要优化
            f.write(text + '\n')

    cursor.close()
    db.close()


if __name__ == "__main__":
    host = "rm-8vbwj6507z6465505ro.mysql.zhangbei.rds.aliyuncs.com"
    user = "root"
    password = "AI@2019@ai"
    database = "stu_db"
    port = 0
    try:
        get_news_from_sql(host, user, password, database, port)
    except Exception:
        # 如果发生异常，则回滚
        print("发生异常", Exception)
        # db.rollback()
        pass
