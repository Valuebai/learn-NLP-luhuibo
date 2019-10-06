#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：LuckyHuibo
@Date   ：2019/10/6 16:19
@Desc   ： 想在linux上定时执行任务，处了linux自带的crontab外，发现了apscheduler和celery
对应的flask框架为：flask-apscheduler

---
python定时任务 apscheduler详细使用教程
官方文档：https://apscheduler.readthedocs.io/en/latest/

---
最近一个程序要用到后台定时任务，看了看python后台任务，一般2个选择，一个是apscheduler，一个celery。

apscheduler比较直观简单一点，就选说说这个库吧。
网上一搜索，晕死，好多写apscheduler的都是超级老的版本，而且博客之间相互乱抄，错误一大堆。
还是自己读官方文档，为大家理一遍吧。
=================================================='''

# coding:utf-8
from apscheduler.schedulers.blocking import BlockingScheduler
import datetime


def aps_test():
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '你好')


scheduler = BlockingScheduler()
scheduler.add_job(func=aps_test, trigger='cron', second='*/5')
scheduler.start()
