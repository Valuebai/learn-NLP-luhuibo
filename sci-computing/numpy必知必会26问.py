#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：LuckyHuibo
@Date   ：2019/7/31 23:42
@Desc   ：numpy必知必会26问
=================================================='''

# numpy必知必会26问
# 1.导入numpy库
import numpy as np

# 2.建立一个一维数组 a 初始化为[4,5,6], (1)输出a 的类型（type）
# (2)输出a的各维度的大小（shape）(3)输出 a的第一个元素（值为4）
a = np.array([4, 5, 6])  # 建立一个一维数组 a 初始化为[4,5,6]
print(type(a))  # 查看数组类型  <class 'numpy.ndarray'>
print(a.shape)  # 输出a的各维度的大小  (3,)
print(a[0])  # 输出 a的第一个元素  4
print("*" * 50)

# 3.建立一个二维数组 b,初始化为 [ [4, 5, 6],[1, 2, 3]] (1)输出各维度的大小（shape）
# (2)输出 b(0,0)，b(0,1),b(1,1) 这三个元素（对应值分别为4,5,2）
b = np.array([[4, 5, 6], [1, 2, 3]])  # 建立一个二维数组 b,初始化为 [ [4, 5, 6],[1, 2, 3]]
print(b.shape)  # 输出各维度的大小  (2, 3)
print(b[0][0], b[0][1], b[0][2])  # 输出 b(0,0)，b(0,1),b(1,1)  4 5 6
print("*" * 50)

# 4. (1)建立一个全0矩阵 a, 大小为 3x3; 类型为整型（提示: dtype = int）(2)建立一个全1矩阵b,大小为4x5;
#   (3)建立一个单位矩阵c ,大小为4x4; (4)生成一个随机数矩阵d,大小为 3x2.
a = np.zeros([3, 3], int)  # 建立一个全0矩阵 a, 大小为 3x3; 类型为整型（提示: dtype = int）
b = np.ones([4, 5])  # 建立一个全1矩阵b,大小为4x5;
c = np.eye(4)  # 建立一个单位矩阵c ,大小为4x4;
d = np.random.rand(3, 2)  # 生成一个随机数矩阵d,大小为 3x2.

# 5. 建立一个数组 a,(值为[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]] ) ,
# (1)打印a; (2)输出 下标为(2,3),(0,0) 这两个数组元素的值
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(a)
print(a[2][3], a[0][0])  # 12 1
print("*" * 50)

# 6.把上一题的 a数组的 0到1行 2到3列，放到b里面去，（此处不需要重新建立a,直接调用即可）
# (1),输出b;(2) 输出b 的（0,0）这个元素的值
row = [i for i in range(0, 2)]  # 定义行
col = [i for i in range(2, 4)]  # 定义列
b = a[row]  # 先取出需要的行
b = b[:, col]  # 再取出需要的列
print(b)
print(b[0][0])  # 3
print("*" * 50)

# 7. 把第5题的 数组的最后两行所有元素放到 c中，（提示：a[1:2][:]）(1)输出 c ;
#   (2) 输出 c 中第一行的最后一个元素（提示，使用 -1 表示最后一个元素）
c = a[1:3][:]  # [1:3] 不包括3
print(c)
print(c[0][-1])  # 8
print("*" * 50)

# 8.建立数组a,初始化a为[[1, 2], [3, 4], [5, 6]]，输出 （0,0）（1,1）（2,0）这三个元素
# （提示： 使用 print(a[[0, 1, 2], [0, 1, 0]]) ）
a = np.array([[1, 2], [3, 4], [5, 6]])
print(a[[0, 1, 2], [0, 1, 0]])  # 通过两个列表定位（0，0），（1，1），（2，0） [1 4 5]
print("*" * 50)

# 9.建立矩阵a ,初始化为[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]，
# 输出(0,0),(1,2),(2,0),(3,1) (提示使用 b = np.array([0, 2, 0, 1]) print(a[np.arange(4), b]))
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
b = np.array([0, 2, 0, 1])  # 定位纵坐标
print(a[np.arange(4), b])  # np.arange(4) 获取横坐标[1，2，3，4]      [ 1  6  7 11]
print("*" * 50)

# 10.对9中输出的那四个元素，每个都加上10，然后从新输出矩阵a.(提示：a[np.arange(4), b] += 10 ）
a[np.arange(4), b] += 10  # 加到对应坐标的元素中
print(a[np.arange(4), b])  # [11 16 17 21]
print("*" * 50)

# array 的数学运算
# 11. 执行 x = np.array([1, 2])，然后输出 x 的数据类型，（答案是 int64）
x = np.array([1, 2], dtype="int64")  # 默认为int32
print(x.dtype)  # int64
print("*" * 50)

# 12.执行 x = np.array([1.0, 2.0]) ，然后输出 x 的数据类洗净（答案是 float64）
x = np.array([1.0, 2.0])
print(x.dtype)  # float64
print("*" * 50)

# 13.执行 x = np.array([[1, 2], [3, 4]], dtype=np.float64) ，
# y = np.array([[5, 6], [7, 8]], dtype=np.float64)，然后输出 x+y ,和 np.add(x,y)
x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.array([[5, 6], [7, 8]], dtype=np.float64)
print(np.add(x, y))
print("*" * 50)

# 14. 利用 13题目中的x,y 输出 x-y 和 np.subtract(x,y)
print(np.subtract(x, y))
print("*" * 50)

# 15. 利用13题目中的x，y 输出 x*y ,和 np.multiply(x, y) 还有 np.dot(x,y),比较差异。然后自己换一个不是方阵的试试。
print(np.multiply(x, y))  # 矩阵点乘，对应元素相乘
print(np.dot(x, y))  # 矩阵乘法
print("*" * 50)

# 16. 利用13题目中的，x,y,输出 x / y .(提示 ： 使用函数 np.divide())
print(np.divide(x, y))
print("*" * 50)

# 17. 利用13题目中的，x,输出 x的 开方。(提示： 使用函数 np.sqrt() )
print(np.sqrt(x))
print("*" * 50)

# 18.利用13题目中的，x,y ,执行 print(x.dot(y)) 和 print(np.dot(x,y))
print(x.dot(y))  # 矩阵乘法
print(np.dot(x, y))  # 矩阵乘法
print("*" * 50)

# 19.利用13题目中的 x,进行求和。（提示：输出三种求和 (1)print(np.sum(x)):
# (2)print(np.sum(x，axis =0 ）); (3)print(np.sum(x,axis = 1))）
print(np.sum(x))  # 计算矩阵x所有元素的和
print(np.sum(x, axis=0))  # 计算矩阵x每列的和
print(np.sum(x, axis=1))  # 计算矩阵x每行的和
print("*" * 50)

# 20.利用13题目中的 x,进行求平均数（提示：输出三种平均数(1)print(np.mean(x))
# (2)print(np.mean(x,axis = 0))(3) print(np.mean(x,axis =1))）
print(np.mean(x))  # 矩阵x所有元素之和/矩阵x元素个数
print(np.mean(x, axis=0))  # 矩阵x每列的平均值
print(np.mean(x, axis=1))  # 矩阵x每行的平均值
print("*" * 50)

# 21.利用13题目中的x，对x 进行矩阵转置，然后输出转置后的结果，（提示：x.T 表示对 x 的转置）
print(x.T)
print("*" * 50)

# 22.利用13题目中的x,求e的指数（提示： 函数 np.exp()）
print(np.exp(x))  # e^x
print("*" * 50)

# 23.利用13题目中的 x,求值最大的下标（提示(1)print(np.argmax(x)) ,
# (2) print(np.argmax(x),axis =0)(3)print(np.argmax(x),axis =1))
print(np.argmax(x))
print(np.argmax(x, axis=0))  # 矩阵x每列对应最大值的索引
print(np.argmax(x, axis=1))  # 矩阵x每行对应最大值的索引
print("*" * 50)

# 24.画图，y=x*x, x = np.arange(0, 100, 0.1) （提示这里用到 matplotlib.pyplot 库）
import matplotlib.pyplot as plt

fig = plt.figure()
ph = fig.add_subplot(2, 2, 1)  # 两行两列的第一个图
x = np.arange(0, 100, 0.1)
y = x * x
ph.plot(x, y)

# 25.画图。画正弦函数和余弦函数， x = np.arange(0, 3 * np.pi, 0.1)
# (提示：这里用到 np.sin() np.cos() 函数和 matplotlib.pyplot 库)
y1 = np.sin(x)
y2 = np.cos(x)
ph2 = fig.add_subplot(2, 2, 2)  # 两行两列的第二个图
ph2.plot(x, y1)
ph3 = fig.add_subplot(2, 2, 3)  # # 两行两列的第三个图
ph3.plot(x, y2)
plt.show()

# 26.附加题.执行下面的语句，解释运算结果，了解 nan 和 inf 的含义 print(0*np.nan)
# print(np.nan == np.nan) print(np.inf > np.nan) print(np.nan - np.nan) print(0.3 == 3***0.1)
print(0 * np.nan)  # numpy空类型值为nan
# np.nan 非空对象，其类型为基本数据类型float,用 i is None-->False；np.isnan(np.nan) --> True
print(np.nan == np.nan)  # False
print(np.inf > np.nan)  # np.inf为无穷大
print(np.nan - np.nan)
print(0.3 == 3 * 0.1)  # 3*0.1 -> 0.30000000000000004 精度问题
