#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：LuckyHuibo
@Date   ：2019/7/21 0:23
@Desc   ：review课程上的代码
=================================================='''
from collections import defaultdict

if __name__ == "__main__":
    # Dynamic Programming Problem
    '''
    我的笔记： 当key不存在时，用普通字典打印不存在的键会报keyError
    使用defaultdict，当key不存在时，返回的是工厂函数（list、set、str）的默认值，
    比如list对应[ ]，str对应的是空字符串，set对应set( )，int对应0
    【参考】https://www.jianshu.com/p/26df28b3bfc8
    '''
    original_price = [1, 5, 8, 9, 10, 17, 17, 20, 24, 30, 35]
    price = defaultdict(int)
    for i, p in enumerate(original_price):
        price[i + 1] = p
    print('打印defaultdict.price：', price)
    print('打印打印defaultdict不存在的键', price[99])

    # Get the max splitting by enumerate
    print(max(1, 2, 3, 4))  # max()函数的使用，找到数中的最大值

    '''
    我的笔记：python中调用函数的时候是：func()有加括号的，也可以把函数当作参数传递(func,arg2)传递是函数的地址，python是面向函数的
    '''


    def example(f, arg):
        return f(arg)


    def add_ten(num):
        return num + 10


    def mul_ten(num):
        return num * 10


    operations = [add_ten, mul_ten]
    for f in operations:
        print(example(f, 100))

    called_time = defaultdict(int)


    def get_call_times(f):
        result = f()
        print('function:{} called once!'.format(f.__name__))
        called_time[f.__name__] += 1

        return result


    def some_function_1():
        print('I am function 1')


    get_call_times(some_function_1)
    print(called_time)  # 打印called_time

    call_time_with_arg = defaultdict(int)


    def r(n):
        # fname = r.__name__
        # call_time_with_arg[(fname,n)] +=1
        return max(
            [price[n] + [r(i) + r(n - i) for i in range(1, n)]]
        )


