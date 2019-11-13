---
title: ML-Linear Regression
tags: python,
author:  [Valuebai](https://github.com/Valuebai/)
---

## Machine Learning入门级算法-线性回归


如果试图改变一些东西，首先应该接受许多东西。
## 导读

算法是人类利用解决实际问题的一种工具。其本质是通过人为设定好的函数，给定X能够输出我们想要的Y。

ok，让我们来看一个简单的回归问题。
## 01


线性回归 LinearRegression
线性回归是对自变量和因变量关系一种建模的回归分析。其基准函数: 
y = θ * x + Σ

- X 为特征值的向量X = (x1, x2, ..., xn)
- θ 为每个特征值的权重(参数) θ = (θ1, θ2, ..., θn)
- y 为每个X对应的标签值(真实值)
- Σ 为误差值

![图(1)](https://www.github.com/Valuebai/my-markdown-img/raw/master/小书匠/1568903016640.png)

在给定特征x和标签y的情况下，求出参数w(权重)的合理值。那么其实我们要去预测的值其实就是h(x):

![图(2)](https://www.github.com/Valuebai/my-markdown-img/raw/master/小书匠/1568903000065.png)

既然如此, 随机误差值Σ的值等于样本的真实值减去样本的预测值h(x)（Σ = y - h(x) 这就是我们的目标函数）。而此时只要求出误差Σ的最小值，即得到参数θ。

在一定的数据量情况下令其服从高斯分布得到样本误差的高斯概率密度函数：

![图(3)](https://www.github.com/Valuebai/my-markdown-img/raw/master/小书匠/1568903056538.png)

随机误差Σ联合概率，得到似然函数如下：

![图(4)](https://www.github.com/Valuebai/my-markdown-img/raw/master/小书匠/1568903102612.png)

对图(4)取对数似然得到J(θ)，考虑在θ取什么值的情况下J(θ)值最大：

![图(5)](https://www.github.com/Valuebai/my-markdown-img/raw/master/小书匠/1568903112679.png)

这里对数似然化简后，得到的J(θ)为最小二乘。此时要求θ为什么值的时候使得J(θ)最小（去掉部分为定值不要考虑，因为化简之前需要求J(θ)最大值，去掉一个符号之后自然需要求其最小值，系数1/2 不去掉是因为后面化简会用到）。

不过此时的问题是一般数据中都会存在异常点，这些异常点会造成模型预测的不准确。如果选择的特征较为复杂的情况下就出现过拟合的情况。那么我们怎么解决呢？

## 02
处理异常值之前，大家可以先想想什么是正则化？正则化又是做什么的呢？


拟合数据时，很容易出现过拟合现象(训练集表现很好，测试集表现较差)，这会导致模型的泛化能力下降，这时候，我们就需要使用正则化，降低模型的复杂度。

正则化的核心思想：
机器学习的过程是一个 通过修改参数 θ 来减小误差的过程。
对于上面得到的成本函数,  L1 L2 就只是在这个误差公式后面多加了一个东西, 
让误差不仅仅取决于拟合数据拟合的好坏, 而且取决于像刚刚  那些参数θ值的大小. 
如果是每个参数的平方, 那么我们称它为 L2正则化, 如果是每个参数的绝对值, 我们称为 L1 正则化。

L1正则化：所有系数的绝对值之和；
![enter description here](https://www.github.com/Valuebai/my-markdown-img/raw/master/小书匠/1568904098188.png)

L2正则化：所有系数的平方之和；

![enter description here](https://www.github.com/Valuebai/my-markdown-img/raw/master/小书匠/1568904235367.png)

这里简单的提一下多项式回归。
多项式回归和线性回归基本类似只不过其基准函数是多项式方程。
y = ax1 + bx2**2 + cx3**3...

注：在使用正则化时通常加入λ系数用来控制正则强度。在模型中需要调节的超参数。

## 03

**优点：**
- 模型简单易于理解、实现方便
- 建模速度快，不需要很复杂的计算，在数据量大的情况下依然运行速度很快
- 可以根据系数给出每个变量的理解和解释

**缺点：**
- 对异常值敏感
- 无法解决多分类问题

**损失函数**
- 线性回归其随时函数为最小平方误差和（最小二乘误差）。
- 线性回归的损失函数在是一个凸函数，能够通过梯度下降得到最优解。

思考：逻辑回归的损失函数为什么不是最小平方和误差？

```python
coding: 回归算法代码



# 使用sklearn.datasets中波士顿房价数据

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_boston
import numpy as np
from sklearn.metrics import r2_score
data = load_boston()
x = data.data
y = data.target
train_x, text_x, train_y, test_y = train_test_split(x, y, random_state=7)
model = LinearRegression()
model.fit(train_x, train_y)
y_hat = model.predict(text_x)
print('r2_score',r2_score(test_y, y_hat))


正则化：

model = Lasso(alpha=1)
model.fit(train_x, train_y)
y_hat = model.predict(text_x)
print('r2_score', r2_score(test_y, y_hat))


L2正则化：

model = Ridge(alpha=1)
model.fit(train_x, train_y)
y_hat = model.predict(text_x)
print('r2_score', r2_score(test_y, y_hat))


注：这里只演示sklearn对回归算法的使用，可能数据的拟合效果并不好。


```


【Me】https://github.com/Valuebai/

【参考】
1、出处：https://mp.weixin.qq.com/s/LR7Z0RE-CZIgHBGyxeg1Xw