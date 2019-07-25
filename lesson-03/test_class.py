#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：LuckyHuibo
@Date   ：2019/7/21 0:23
@Desc   ：review课程上的代码
=================================================='''
from collections import defaultdict
from functools import wraps
import pysnooper
from functools import lru_cache

# @lru_cache这个装饰器实现了备忘的功能，是一项优化技术，把耗时的函数的结果保存起来，避免传入相同的参数时重复计算

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


    # =================================
    # 进行python面向函数的练习，将函数名作为参数
    # =================================
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

    # end
    # =================================
    # 装饰器的练习
    # =================================

    original_price = [1, 5, 8, 9, 10, 17, 17, 20, 24, 30]
    price = defaultdict(int)
    solution = {}

    for i, p in enumerate(original_price):
        price[i + 1] = p

    print('original_price is :', price)


    def memo(func):
        cache = {}

        @wraps(func)  # functools.wraps(func)装饰器的作用是将func函数的相关属性复制到被装饰的函数
        def _wrap(n):  # 通常用*args,**kwargs
            if n in cache:
                result = cache[n]
            else:
                result = func(n)
                cache[n] = result
            return result

        return _wrap


    @memo
    def r_d(n):
        max_price, max_split = max(
            [(price[n], 0)] + [(r(i) + r(n - i), i) for i in range(1, n)], key=lambda x: x[0]
        )
        solution[n] = (n - max_split, max_split)
        print('n=', n, [(price[n], 0)] + [(r(i) + r(n - i), i) for i in range(1, n)])
        return max_price


    def r(n):
        # fname = r.__name__
        # call_time_with_arg[(fname,n)] +=1
        return max(
            [price[n]] + [r(i) + r(n - i) for i in range(1, n)]
        )


    print('有装饰器, max=', r_d(13))
    print('没有装饰器, max=', r(3))


    # end
    # =================================
    # 编辑距离的练习
    # =================================
    # 边界距离
    @pysnooper.snoop(r'logs.log')
    def Levenshtein_Distance(str1, str2):
        """
        计算字符串 str1 和str2的编辑距离
        :param str1:
        :param str2:
        :return:
        """
        matrix = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
        print('matrix is', matrix)
        for i in range(1, len(str1) + 1):
            for j in range(1, len(str2) + 1):
                if (str1[i - 1] == str2[j - 1]):
                    d = 0
                else:
                    d = 1

                matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)

        return matrix[len(str1)][len(str2)]


    print('Levenshtein_Distance=', Levenshtein_Distance('ABC', 'BCDef'))

    # @lru_cache这个装饰器实现了备忘的功能，是一项优化技术，把耗时的函数的结果保存起来，避免传入相同的参数时重复计算
    from functools import lru_cache

    solution = {}


    @lru_cache(maxsize=2 ** 10)
    @pysnooper.snoop(r'logs.log')
    def edit_distance(string1, string2):

        if len(string1) == 0: return len(string2)
        if len(string2) == 0: return len(string1)

        tail_s1 = string1[-1]
        tail_s2 = string2[-1]

        if tail_s1 == tail_s2:
            both_forward = (edit_distance(string1[:-1], string2[:-1]) + 0, '')
        else:
            both_forward = (edit_distance(string1[:-1], string2[:-1]) + 1, 'SUB {} => {}'.format(tail_s1, tail_s2))

        candidates = [
            (edit_distance(string1[:-1], string2) + 1, 'DEL {} '.format(tail_s1)),  # string 1 delete tail
            (edit_distance(string1, string2[:-1]) + 1, 'ADD {} '.format(tail_s2)),  # string 1 add tail of string2
        ]
        candidates.append(both_forward)

        min_distance, operation = min(candidates, key=lambda x: x[0])

        solution[(string1, string2)] = operation
        return min_distance


    print('edit_distance=', edit_distance('BC', 'ABCD'))
    print('solution=', solution)
