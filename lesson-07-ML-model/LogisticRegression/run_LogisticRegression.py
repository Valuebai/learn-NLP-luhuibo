#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：Valuebai
@Date   ：2019/11/21 18:11
@Desc   ：
逻辑回归实际上是一种二分类回归，生活中有许多事例都能用逻辑回归解释。例如大和小，少和多，高和低等。对于多分类其实可以用多个二分类去解释。

本次案描述了研发成本和产品是否合格之间的关系

逻辑回归
1 建立数据集
2 提取特征标签
3 绘制散点图
4 建立训练数据和测试数据
5 训练模型（使用训练数据）
6 模型评估
7 进一步理解什么是逻辑函数

=================================================='''

# 1  建立数据集

from collections import OrderedDict
import pandas as  pd

R_costDic = {'研发成本': [50, 80, 150, 200, 280, 400, 500, 600], '产品质量': [0, 0, 0, 0, 1, 1, 1, 1]}
# 0表示不合格， 1表示合格
R_costOD = OrderedDict(R_costDic)
R_costDf = pd.DataFrame(R_costOD)
print('R_costDf.head() is {0}'.format(R_costDf.head()))

# 2  提取特征标签
# 特征features
R_X = R_costDf.loc[:, '研发成本']
# 标签labes
R_y = R_costDf.loc[:, '产品质量']

# 3  绘制散点图

import matplotlib.pyplot as plt

plt.figure(figsize=(6, 5))
plt.scatter(R_X, R_y, color="b", label="R_cost data")
plt.xlabel('R_cost')
plt.ylabel('quality')
plt.show()

#  4  建立训练数据和测试数据

from sklearn.model_selection import train_test_split

# 建立训练数据和测试数据
X_train, X_test, y_train, y_test = train_test_split(R_X,
                                                    R_y,
                                                    train_size=.8)
# 输出数据大小
print('原始数据特征：', R_X.shape, '，训练数据特征：', X_train.shape, '，测试数据特征：', X_test.shape)
print('原始数据标签：', R_y.shape, '训练数据标签：', y_train.shape, '测试数据标签：', y_test.shape)

# 散点图
plt.scatter(X_train, y_train, color="blue", label="train data")
plt.scatter(X_test, y_test, color="red", label="test data")

# 添加图标标签
plt.legend(loc='lower right')
plt.xlabel("Hours")
plt.ylabel("Pass")
# 显示图像
plt.show()

# 5  训练模型（使用训练数据）

# 将训练数据特征转换成二维数组XX行*1列
X_train = X_train.values.reshape(-1, 1)
# 将测试数据特征转换成二维数组行数*1列
X_test = X_test.values.reshape(-1, 1)

# 导入逻辑回归包
from sklearn.linear_model import LogisticRegression

# 创建模型：逻辑回归
model = LogisticRegression()
# 训练模型
model.fit(X_train, y_train)

# 6  模型评估

# 评估模型：准确率
model.score(X_test, y_test)

# 7  进一步理解什么是逻辑函数
# 第1个值是标签为0的概率值，第2个值是标签为1的概率值 # array([[0.18851031, 0.81148969]])
model.predict_proba([[500]])

# 预测数据：使用模型的predict方法可以进行预测。输出结果为1表示合格。
pred = model.predict([[500]])
print(pred)

'''
理解逻辑回归函数
斜率slope
截距intercept
'''
# 第1步：得到回归方程的z值
# 回归方程：z=𝑎+𝑏x
# 截距
import numpy as np

a = model.intercept_
# 回归系数
b = model.coef_

x = 500
z = a + b * x
# 第2步：将z值带入逻辑回归函数中，得到概率值
y_pred = 1 / (1 + np.exp(-z))
print('预测的概率值：', y_pred)
