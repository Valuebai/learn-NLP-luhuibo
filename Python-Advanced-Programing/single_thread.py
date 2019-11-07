#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：LuckyHuibo
@Date   ：2019/10/27 9:45
@Desc   ：

Total time:12.383874416351318
=================================================='''
from threading import Thread
import time

I = 0


def my_count():
    global I
    while I <= 100000000:
        I = I + 1


def my_count2():
    """
    单线程
    :return:
    """
    i = 0
    for _ in range(100000000):
        i = i + 1
    return


def main():
    start_time = time.time()
    # t = Thread(target=my_count)
    # t.start()
    # t.join()
    my_count()
    my_count()
    end_time = time.time()
    print(f"Total time:{end_time - start_time}")


if __name__ == "__main__":
    main()
