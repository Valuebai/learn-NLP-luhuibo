#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：LuckyHuibo
@Date   ：2019/7/25 18:27
@Desc   ：
=================================================='''

import random
import numpy as np
import matplotlib.pylab as plt

# 生成20个样本点
X = [random.randint(-100, 100) for i in range(20)]
Y = [random.randint(-100, 100) for _ in range(20)]
# 构成坐标
points = [(x, y) for x, y in zip(X, Y)]
single_point = points.pop(random.randint(0, 19))


def O_distance(point1, point2):
    """计算两个点的距离，这里没有进行开方计算"""
    return np.sqrt(np.square(point1[0] - point2[0]) + np.square(point1[1] - point2[1]))


def min_dist(point, points):
    """寻找所有点中距离当前的点距离最近的点"""
    dist, index = 999999999, None
    for i, p in enumerate(points):
        d = O_distance(p, point)
        if dist > d:
            dist = d
            index = i
    return index, dist


def total_dist(n_point, points):
    """
    有三(n_point)辆车，二十个地点(points)，怎么走才能使得三辆车经过所有的地点，并且路线最短。
    贪心思想：每次选择最近的点。
    :param n_point: 出发点
    :param points: 点集合
    :return: 经过所有点的距离
    """
    from_point = n_point
    sum_dist = 0
    while points:
        for i, point in enumerate(from_point):
            # 计算距离当前车辆最近的点
            index, dist = min_dist(point, points)
            # 车从当前的点转移到最近的点
            from_point[i] = points.pop(index)
            # 计算走过的距离
            sum_dist += dist
    return sum_dist


print(total_dist([single_point], points))
