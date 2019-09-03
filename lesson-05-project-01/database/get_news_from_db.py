#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：LuckyHuibo
@Date   ：2019/8/27 23:03
@Desc   ：
经验：写入txt文本时，千万不要使用range()，会特别的慢！！！

待优化：
1. 目前是获取所有数据保存到txt中，后面写个函数，每次获取1条数据，训练时每次取1条，对数据库压力小点
2. 保存说的相似词到数据库中
=================================================='''
import pymysql, re, logging
from config.get_ai_db import GetConfParams

ConfParams = GetConfParams()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(lineno)d -  %(message)s')
logger = logging.getLogger(__name__)


def connect_db(host, port, user, password, db):
    """
    连接数据库并获取数据
    :param host: 主机
    :param port: 端口
    :param user: 用户名
    :param password: 密码
    :param db: 数据库名
    连接时，默认字符集charset='utf8'，不然会有乱码
    :return:
    """
    logging.info('Connect database...')
    conn = pymysql.connect(host=host, port=port, user=user, password=password, db=db, charset='utf8')  # 数据库的链接
    cur = conn.cursor()  # 获取一个游标
    logging.info('get data from table...')
    cur.execute("select content from news_chinese")  # 具体的数据库操作语句
    contents = cur.fetchall()  # 将所有查询结果返回为元组
    print(len(contents))
    cur.close()  # 关闭游标
    conn.close()  # 释放数据库资源
    logging.info('Close Connect database...')
    return contents


def save(contents):
    """
    获取并保存新闻文本
    :param contents: News from db
    :return:
    """
    with open('../data/news.txt', 'w', encoding='utf-8') as f:
        for content in contents:
            content = clean(content[0])
            logger.info('Start saving.......')
            f.write(content + '\n')


def clean(s):
    """
    清洗数据
    :param s: 文本
    :return:
    """
    re_compile = re.compile(r'�|《|》|\/|）|（|【|】|\\n|\\r|\\t|\\u3000|;|\*')
    string = re_compile.sub('', str(s))
    return string


if __name__ == "__main__":
    # 从config配置文件中读取数据库信息
    host = ConfParams.host
    user = ConfParams.user
    password = ConfParams.password
    database = ConfParams.db_name
    port = ConfParams.port

    # connect db and get the data
    news = connect_db(host, port, user, password, database)
    # save the data in txt
    save(news)
