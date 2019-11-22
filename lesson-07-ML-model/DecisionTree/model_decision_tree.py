#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：Valuebai
@Date   ：2019/11/14 23:18
@Desc   ：机器学习之分类与回归树(CART)

CART( Classification And Regression Tree) Sklearn实现
我们以sklearn中iris数据作为训练集，
iris属性特征包括花萼长度、花萼宽度、花瓣长度、花瓣宽度，
类别共三类，分别为Setosa、Versicolour、Virginca。


CART决策树的sklearn实现及其GraphViz可视化
https://blog.csdn.net/chai_zheng/article/details/78226556
=================================================='''

from sklearn.datasets import load_iris
from sklearn import tree
from matplotlib import pyplot as plt
iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

tree.plot_tree(clf.fit(iris.data, iris.target))
plt.show()
