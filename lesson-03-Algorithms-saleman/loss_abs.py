#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：LuckyHuibo
@Date   ：2019/7/25 17:55
@Desc   ：use the absolution to count the loss
according the code given in class, change the loss
to abs, and try the differences.
=================================================='''
import pysnooper
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston


# defined the func of y
def y_func(age, k, b):
    return k * age + b


# defined the func of loss
def loss(y, yhat):
    return np.mean(np.abs(y - yhat))


# count the derivate_k
def derivate_k(y, yhat, x):
    abs_values = [1 if (y_i - yhat_i) > 0 else -1 for y_i, yhat_i in zip(y, yhat)]
    return np.mean([a * -x_i for a, x_i in zip(abs_values, x)])


# count the derivate_b
def derivate_b(y, yhat):
    abs_values = [1 if (y_i - yhat_i) > 0 else -1 for y_i, yhat_i in zip(y, yhat)]
    return np.mean([a * -1 for a in abs_values])


if __name__ == "__main__":
    # realize the data
    boston = load_boston()
    print(boston.data.shape)
    X, y = boston['data'], boston['target']

    price = y
    room_num = X[:, 5]
    k_hat = random.random() * 20 - 10
    b_hat = random.random() * 20 - 10
    best_k, best_b = k_hat, b_hat
    # e 科学计数法，e-1 = 0.1, e-2 = 0.01
    learnin_rate = 1e-1
    # 循环的次数
    loop_times = 1
    # loss list
    losses = []
    while loop_times < 10000:
        k_delta = -1 * learnin_rate * derivate_k(price, y_func(room_num, k_hat, b_hat), room_num)
        b_delta = -1 * learnin_rate * derivate_b(price, y_func(room_num, k_hat, b_hat))

        k_hat += k_delta
        b_hat += b_delta

        estimated_price = y_func(room_num, k_hat, b_hat)
        error_rate = loss(y=price, yhat=estimated_price)

        print('loop =={}'.format(loop_times), end='')
        print(' f(age) = {} * age + {},with error rate: {}'.format(best_k, best_b, error_rate))

        losses.append(error_rate)
        loop_times += 1

    plt.scatter(X[:, 5], y)
    #plt.plot(range(len(losses)), losses)
    plt.scatter(room_num,estimated_price)
    plt.show()
