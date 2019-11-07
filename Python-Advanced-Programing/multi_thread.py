#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：LuckyHuibo
@Date   ：2019/10/27 9:45
@Desc   ：演示多线程比单线程运算速度慢，因为在计算过程中频繁地进行切换，耗时更多


Total time:13.924792289733887
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
    多线程
    :return:
    """
    i = 0
    for _ in range(100000000):
        i = i + 1
    return


def main():
    start_time = time.time()
    threads = []
    for i in range(2):
        t = Thread(target=my_count)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
    end_time = time.time()
    print(f"Total time:{end_time - start_time}")


if __name__ == "__main__":
    main()
