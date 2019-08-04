#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：LuckyHuibo
@Date   ：2019/8/5 0:18
@Desc   ：softmax的学习及实现

softmax函数将任意n维的实值向量转换为取值范围在(0,1)之间的n维实值向量，并且总和为1。
例如：向量softmax([1.0, 2.0, 3.0]) ------> [0.09003057, 0.24472847, 0.66524096]

性质：

1. 因为softmax是单调递增函数，因此不改变原始数据的大小顺序。
2. 将原始输入映射到(0,1)区间，并且总和为1，常用于表征概率。
3. softmax(x) = softmax(x+c), 这个性质用于保证数值的稳定性。

softmax的实现及数值稳定性


import numpy as np


def softmax(x):
    """Compute the softmax of vector x."""
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


# 让我们来测试一下上面的代码：

print(softmax([1, 2, 3]))
# result: array([0.09003057, 0.24472847, 0.66524096])

# 但是，当我们尝试输入一个比较大的数值向量时，就会出错：

print(softmax([1000, 2000, 3000]))
# result: array([nan, nan, nan])

这是由numpy中的浮点型数值范围限制所导致的。
当输入一个较大的数值时，sofmax函数将会超出限制，导致出错。
为了解决这一问题，这时我们就能用到sofmax的第三个性质，
即：softmax(x) = softmax(x+c)，

一般在实际运用中，通常设定c = - max(x)。
接下来，我们重新定义softmax函数：
=================================================='''

import numpy as np


def softmax(vec):
    """Compute the softmax in a numerically stable way."""
    vec = vec - np.max(vec)  # softmax(x) = softmax(x+c)
    exp_x = np.exp(vec)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


# 然后再次测试一下：

print(softmax([1000, 2000, 3000]))
# result: array([0., 0., 1.])
# Done!


# 以上都是基于向量上的softmax实现，下面提供了基于向量以及矩阵的softmax实现，代码如下：

import numpy as np


def softmax_matrix(x):
    """
    Compute the softmax function for each row of the input x.

    Arguments:
    x -- A N dimensional vector or M x N dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        exp_minmax = lambda x: np.exp(x - np.max(x))
        denom = lambda x: 1.0 / np.sum(x)
        x = np.apply_along_axis(exp_minmax, 1, x)
        denominator = np.apply_along_axis(denom, 1, x)

        if len(denominator.shape) == 1:
            denominator = denominator.reshape((denominator.shape[0], 1))

        x = x * denominator
    else:
        # Vector
        x_max = np.max(x)
        x = x - x_max
        numerator = np.exp(x)
        denominator = 1.0 / np.sum(numerator)
        x = numerator.dot(denominator)

    assert x.shape == orig_shape
    return x
