#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：LuckyHuibo
@Date   ：2019/8/26 18:38
@Desc   ：
=================================================='''
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.automap import automap_base


def get_database(host='rm-8vbwj6507z6465505ro.mysql.zhangbei.rds.aliyuncs.com',
                 db_name='stu_db', user='root', password='AI@2019@ai', port='3306'):
    """
    create existed postgre database engine and get its table with sqlalchemy
    :param host:host name, eg:'rm-8vbwj6507z6465505ro.mysql.zhangbei.rds.aliyuncs.com'
    :param db_name: mysql database's name,eg:'stu_db'
    :param user: mysql database user's name, eg: 'root'
    :param password: mysql databse user's password: AI@2019@ai
    :return: db_engine, database engine,can used by sqlalchemy or pandas
             tables, database's tables
    """
    db_type = "mysql+mysqlconnector"  # mysql+mysqlconnector
    string = "%s://%s:%s@%s:%s/%s" % (db_type, user, password, host, port, db_name)
    db_engine = create_engine(string, echo=False, encoding='utf-8')
    # get sqlalchemy tables from database
    Base = automap_base()
    Base.prepare(db_engine, reflect=True)
    tables = Base.classes
    return db_engine, tables


def get_table(table_name='news_chinese'):
    # create a session with database
    session = sessionmaker(bind=db_engine)()

    query_news = session.query(tb_name).all()
    with open(table_name + '.txt', 'w', encoding='utf-8') as f:
        for article in query_news:
            f.write(str(article.content) + '\n')
    return


if __name__ == '__main__':
    db_engine, db_table = get_database()

    table_name = 'news_chinese'
    tb_name = db_table[table_name]

    result = get_table()
