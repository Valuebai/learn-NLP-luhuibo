# AI&NLP
## 记录学习NLP之路，一起加油 learn NLP together, code for AI


![Languages](https://img.shields.io/badge/Languages-Python3.6-green)
![Build](https://img.shields.io/badge/Build-passing-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-orange.svg)
![Contributions](https://img.shields.io/badge/Contributions-welcome-ff69b4.svg)


- lesson的命名跟上课内容有些有出入，目前的命名是按照我的学习思路来填充


## Books
- [记录在Books/readme.md](https://github.com/Valuebai/learn-NLP-luhuibo/tree/master/Books)

## 思维导图
- [思维导图]()

## 面试题目
- [面试题目]()


---
## lesson-01 知识点
- 人工智能的历史、发展、应用的概述

**[ML神器：sklearn的快速使用](https://www.cnblogs.com/lianyingteng/p/7811126.html)**
> 传统的机器学习任务从开始到建模的一般流程是：获取数据 -> 数据预处理 -> 训练建模 -> 模型评估 -> 预测，分类。

> [特征工程：预处理数据的方法总结（使用sklearn-preprocessing](https://blog.csdn.net/weixin_40807247/article/details/82793220)




## lesson-02 知识点
- 待补充
- 爬虫获取广州地铁路线的数据
- 利用广度/深度优先搜索算法，获取2个地铁站之间的换乘路线
- 因为百度的页面太不规范，导致有几个页面无法获取数据，故我使用的是WIKI百科的，https://zh.wikipedia.org/wiki/%E5%B9%BF%E5%B7%9E%E5%9C%B0%E9%93%81    
- 使用高德地图API获取坐标：https://map.amap.com/subway/index.html，根据坐标点绘制广州地铁图
![image](https://user-images.githubusercontent.com/9695113/65223170-2d188300-daf3-11e9-91a7-0b3a2c7bd077.png)
- 使用的技术：爬虫-beautifulsoup-数据处理, re, bfs, dfs等




---
## lesson-03 知识点
- LR线性回归问题，尝试不同loss实现梯度下降
- 编辑距离问题
  - 使用动态规划思想来实现编辑距离求解问题
- 从某点出发，经过所有点，要求整个路程最短
  - 使用动态规划解决1人出发问题




---
## lesson-04 知识点
- 使用维基百科下载中文语料库
- python wikipedia extractor抽取维基百科的内容
- jieba分词
- 简繁体转换
- softmax()函数
- 词向量
- one-hot-code
- word2vec(CBOW, skip-gram)
- gensim.models.word2vec进行词向量训练
- Sklearn中TSNE进行词向量的可视化

#### 面试题：如果实现softmax()，代码是？为什么要用它？




---
## lesson-05-project-01 知识点

**NLP common Tools**

- NER
- TFIDF
- wordcloud
- 用jieba 词性——解决80%左右的问题
- jieba, pyltp,coreNLP 三个的比较

- pyltp 中文分词的使用

**面试题：为什么要用cos距离，夹角的距离来计算文本的相似度呢？**

- Project-01[新闻任务言论自动提取]已整理，详情见：https://github.com/Valuebai/NewsInfo-Auto-Extration

![pyltp简介](./images/pyltp简介.jpg)




---

## lesson-06 知识点 

### 主要讲【机器学习】相关的基础知识

> [Machine Learning-线性回归算法分析](https://mp.weixin.qq.com/s/JeMjUzEW0qarQObKTgn8lg) 
> [Machine Learning 2 - 非线性回归算法分析](https://mp.weixin.qq.com/s/BsWKlCrtV6TPOS6S1zS_1w)

### 机器学习的评价标准

- 准确率Accuracy
- 错误率Error
- 精确率Precision
- 召回率Recall
- 均方根误差RMSE, Root Mean Square Error

- F1 score
- P-R曲线
- ROC曲线
- AUC计算

- 过拟合
- 欠拟合


### 面试题A：为什么叫朴素贝叶斯？？
待补充


### 面试题B：什么是欠拟合，什么是过拟合？在项目中有没有遇到过，是怎么解决的呢？
- 欠拟合：模型在训练和预测时表现都不好
- 过拟合：模型在训练集上的表现很好，但在测试集和新数据上的表现较差。
（用数据训练出来的A，在实际应用B上不准或不对）

作业：总结下overfitting（欠拟合）和 underfitting（过拟合）的原因(也是解决的办法，在下面)


### 面试题C：什么是Bias（偏差）和 Vaiance（方差）？，统计学背景的面试官可能会问

[Bias(偏差)，Error(误差)，Variance(方差)及CV(交叉验证)](https://www.jianshu.com/p/8d01ac406b40)


```md
# 摘自：忘了出处的总结
**过拟合**
当模型相对于训练数据的数量和噪度都过于复杂时，会发生过度拟合。
可能的解决方案如下：
·简化模型：可以选择较少参数的模型（例如，选择线性模型而不是高阶多项式模型），可以减少训练数据中的属性数量，又或者是约束模型。
·收集更多的训练数据。
·减少训练数据中的噪声（例如，修复数据错误和消除异常值）。通过约束模型使其更简单，并降低过度拟合的风险，这个过程称为正则化。

**欠拟合**

训练数据拟合不足
你可能已经猜到了，拟合不足和过度拟合正好相反：它的产生通常是因为，对于下层的数据结构来说，你的模型太过简单。
举个例子，用线性模型来描述生活满意度就属于拟合不足；现实情况远比模型复杂得多，所以即便是对于用来训练的示例，该模型产生的预测都一定是不准确的。
解决这个问题的主要方式有：
·选择一个带有更多参数、更强大的模型
·给学习算法提供更好的特征集（特征工程）
·减少模型中的约束（比如，减少正则化超参数）
```

```md
# 摘自：《百面机器学习》，结合上面的，更好理解
# 降低**过拟合**风险的方法
- 从数据入手，获得更多的训练数据
- 降低模型复杂度
- 正则化方法
- 集成学习方法(多个模型集成在一块)

# 降低**欠拟合**风险的方法
- 添加新特征
- 增加模型复杂度
- 减小正则化系数

```

*正则化的理解*
> [正则化的核心思想、L1/L2正则化](https://mp.weixin.qq.com/s/LR7Z0RE-CZIgHBGyxeg1Xw)

```
对于线性方程的求解，是属于线性代数的范畴。
首先要判断是否有解及是否有唯一解；其次具体求解方法则有矩阵消元法，克莱姆法则，逆矩阵及增广矩阵法等等。
对于大多数给定数据集，线性方程有唯一解的概率比较小，多数都是解不存在的超定方程组。
对于这种问题，在计算数学中通常将参数求解问题退化为求最小误差问题，找到一个最接近的解，即术语松弛求解。
```

> 基于均方误差最小化进行模型求解的方法称为“最小二乘法”（least square method），即通过最小化误差的平方和寻找数据的最佳函数匹配；

---
## lesson-07

**课程笔记**: 详情见lesson-07下的readme

- 目标：判别一篇新闻的来源是否为新华社（新闻文章抄袭判断）
- 重点：样本中有 87% 来源是新华社，因此低于87%的的判别可以认为是无效的
- 方法：TFIDF向量化
- 评估模型效果并利用模型进行预测：Precision； Recall； F1 Score ；ROC AUC score
- 利用逻辑回归KNN等方法构建模型，
Logistic Regression、KNN、SVM、Naive Bayes、Random Tree、 Random Forest

**Assignmet：lesson-07&lesson-08的【小项目】文本相似性检测与抄袭判断**

---
## lesson-08知识点 

**课程笔记**: 详情见lesson-08下的readme

基于新闻文章抄袭判断来测试各类模型算法
- KNN
- 逻辑回归
- 决策树
- 朴素贝叶斯
- SVM支持向量机
- 随机森林


**在做所有机器学习问题时：Important of Preprocessing**

- Balance Analysis, 确定一个基准Base line
- Remove Noise, outliers问题
- Remove Collinearity, 数据越大，纬度越大（纬度灾难）
- Rescale Inputs, lesson-07提到


```
老师，发一下你之前讲的SVM推导的视频吧
Minchiuan Gao对所有人说： 09:11 PM
https://zoom.us/recording/share/j5O4MP9x5eIBYl-CFbBBSCf8ySTaCy_Zt3BRDIocJRawIumekTziMw
密码： AI@NLP Xin对所有人说： 09:12 PM 显示密码错误 Minchiuan Gao对所有人说： 09:12 PM A1@NLP
```

**笔记：学习这几个模型，要学会科学的思维**
- 怎么调参数呢？——1. 观察    2. 猜想   3. 动手实践    4. 检查与验证

**面试题：**
1. 为什么叫SVM？
2. 什么叫核函数
3. SVM的空间变换
4. KNN-速度慢，空间大
5. 为什么叫朴素贝叶斯，PPT的计算题
6. K-means的时间复杂度是多少？（阿里内部还是有用到K-means的）

> 【小项目】新华社新闻抄袭自动判别
```
任务要求：

1. 构建一个机器学习模型，判断这个文章是不是新华社的
2. 当这个模型的acc 大于 0.8778， recall， precision，f1等值都较高的时候
3. 用该模型 判断一篇文章是否是新华社的文章，如果判断出来是新华社的，但是，它的source并不是新华社的，那么，我们就说，这个文章是抄袭的新华社的文章
4. Text Representation uses "https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html"
```



---
## lesson-09知识点 

**课程笔记**: 详情见lesson-09下的readme

<font color=red>**本次作业，你需要完成1, 2, 3三个联系。 能够掌握：**</font>
- 数据预处理的知识
- 深度学习程序的构建方式
- 训练集、测试集、验证集
- 神经网络的基础知识
- 图计算网络

- 正则化
- 等非常重要的知识

<font color=red>**assignment**</font>



- 1_notmnist 训练
- 2_fullyconnected
- 3_regularization ——《统计学习方法》书中有很好的介绍，待总结

---
## lesson-10知识点
- 课堂笔记，详情见：lesson-10下的readme


课程内容
- tensorflow图计算
- pytorch和keras等神经网络的原理

作业
- 手把手搭建NN
- 用word2vec+fully connected neural networks 完成“豆瓣电影评论”分类

Reading
TensorFlow-Examples：https://github.com/aymericdamien/TensorFlow-Examples

---
## lesson-11-Project-02知识点

- 【高老师】构建一个简单神经网络框架，手写NN
- Tensorflow练习
- 【小项目】电影评论分类

- 待阅读：论文！！！
- 课堂笔记，详情见：lesson-11下的readme

- Project-02已整理，详情见：https://github.com/Valuebai/Text-Auto-Summarization




---
## lesson-12知识点

**课程笔记**: 详情见lesson-12下的readme
- CNN分类图片
- 从LetNet到ResNet总结
1. Project II
2. Convolutional Neural Networks
3. Pooling, Batch Normalization
4. Transfer Learning

**作业**：详情见lesson-12下的readme.md
Tensorflow CNN
1. https://github.com/Computing-Intelligence/
jupyters_and_slides/blob/master/2019-summer/
assignments/assignment-12-cnn.ipynb
2. 列出从LeNet到ResNet的⽹络的变化和区别

**面试题：** 
- CV课程老师的视频


---
## lesson-13知识点

**课程笔记**: 详情见lesson-13下的readme.md
- RNN 及变体LSTM, GRU
- RNN in pure python, Tensorflow, Keras
- Some Useful applications for RNN
- Kaggle使用
- Toxic Comment Classification Challeng
  - 按照readme下的Baseline1，2,3 不同模型训练+数据处理+提分
- fastText

**作业**：详情见lesson-13下的readme.md
- Reading: 
  - 1. Hands-on tensorflow and scikit learning 
  - 2. https://colah.github.io/posts/2015-08-Understanding-LSTMs/ 
  - 3. https://arxiv.org/pdf/1506.00019.pdf 
  - 4. https://arxiv.org/pdf/1409.3215.pdf
  - 5. https://www.kaggle.com/suniliitb96/tutorial-keras-transfer-learning-withresnet50/notebook
- Kaggle Project: https://www.kaggle.com/c/jigsaw-toxic-comment-classificationchallenge
  - 找到⼀个notebook解决⽅案，修改源代码，使你的结果超过原始作者
  - 然后使⽤修改过的解决⽅法，解决⾖瓣评分的问题
  - 把⾃⼰的源代码都发布到GitHub上

**面试题：**
1. LSTM如何防止梯度爆炸
2. LSTM视频中的图，要非常熟悉

- 工作中的效果达不到领导的要求——说训练数据不够




## lesson-14-Project-03知识点

**课程笔记**: 课堂笔记，详情见lesson-14下的readme
1. 复习 Word2Vec
2. Sequence to Sequence (Seq2Seq)
3. CoVe, EMLo, FastText, SQUAD
4. 参考文献
5. 关于项目3 

**作业**：详情见lesson-14下的readme.md
1. Seq2Seq
2. Github Source Code
https://github.com/fortyMiles/neural-machine-translation
3. https://arxiv.org/abs/1706.03762
4. http://papers.nips.cc/paper/7209-learned-in-translation-contextualized-word-vectors.pdf
5. https://allennlp.org/elmo
6. https://github.com/tensorflow/tensor2tensor

一文看懂NLP：word 表示技术的变迁，从word2vec-elmo-GPT-BERT
> https://tech.yiewan.com/news-id-3408.html

**Project-03已整理**，详情见：https://github.com/Valuebai/Sentimental-Analysis-of-Dianping


---
## lesson-15知识点

- Dialogue
- 课堂笔记，详情见：lesson-15下的readme

---
## lesson-16-Project-04知识点

- Project-04已整理，详情见：
- 课堂笔记，详情见：lesson-16下的readme


---
## QA

- [课程答疑记录，详情见QA/readme.md](https://github.com/Valuebai/learn-NLP-luhuibo/tree/master/QA)


---
## Sci-Computing

- [详情见./Sci-computing/readme.md](https://github.com/Valuebai/learn-NLP-luhuibo/tree/master/sci-computing)



---
## Tensorflow

- [Tensorflow相关学习/readme.md](https://github.com/Valuebai/learn-NLP-luhuibo/tree/master/tensorflow)

---
## Python进阶
- python代码一定多练啊！手敲别人代码也行啊！！不会就敲多做笔记！！！


---
## requirements.txt
- 生成指南：
- 第一步：安装包 pip install pipreqs
- 第二步：在对应路径cmd，输入命令生成 requirements.txt文件：pipreqs ./ --encoding=utf8 避免中文路径报错
- 第三步：下载该代码后直接pip install -r requirements.txt

## 检查python的语法正确性
- 第一步：安装包 pip install --upgrade pyflakes
- 第二步：用这个命令（pyflakes xxx.py）就能检查xxx.py文件中的错误。




## 其他公开课

11月5日公开课《数据挖掘概论与实践》视频和课件链接：
https://pan.baidu.com/s/1tV3TXhP8SB8_ik65ECDQaA 
提取码：2onn 

11.14日大数据开发公开课：
电商双十一背后的实时计算核心技术之Spark Streaming——视频、课件：
链接：https://pan.baidu.com/s/1k5-TY156crmMpqSM0Jc_iQ 
提取码：2sqx 

11月14日公开课，数据分析之kNN(k近邻)算法精讲课件及回放视频网盘链接：https://pan.baidu.com/s/1U5UWnqbSHiB203wBM88giw 
提取码：v2nk 


## BAT 常见的面试题（不分先后）

自我介绍/项目介绍
类别不均衡如何处理
数据标准化有哪些方法/正则化如何实现 /onehot 原理
为什么 XGB 比 GBDT 好
数据清洗的方法有哪些/数据清洗步骤
缺失值填充方式有哪些
变量筛选有哪些方法
信息增益的计算公式
样本量很少情况下如何建模
交叉检验的实现
决策树如何剪枝
WOE/IV 值计算公式
分箱有哪些方法/分箱原理是什么
手推 SVM：目标函数，计算逻辑，公式都写出来，平面与非平面
核函数有哪些
XGB 原理介绍/参数介绍/决策树原理介绍/决策树的优点
Linux/C/Java熟 悉程度
过拟合如何解决
平时通过什么渠道学习机器学习（好问题值得好好准备）
决策树先剪枝还是后剪枝好
损失函数有哪些
偏向做数据挖掘还是算法研究（好问题）
bagging 与 boosting 的区别
模型评估指标有哪些
解释模型复杂度/模型复杂度与什么有关
说出一个聚类算法
ROC 计算逻辑
如何判断一个模型中的变量太多
决策树与其他模型的损失函数、复杂度的比较
决策树能否有非数值型变量
决策树与神经网络的区别与优缺点对比
数据结构有哪些
model ensembling 的方法有哪些


## 如何在 Jupyter 中写数学公式

```jupyter
**插入**：我们在这里快速的过一下如何在 Jupyter 中写数学公式。 这个其实很简单，如果我们要输入一个公式，例如A1, 那么，我们在Jupyter中输入`$A_i$`, 然后 Enter， 是不是就变成了$A_i$? 其实两个`$$`之间的东西就是 Latex 的符号，`$..$`这个我们叫做inline模式，意思就是说你写出来的公式是和你的文字在一行里，如果你`$$..$$``，这个公式就会单独是一行。

我们现在再试一个, 输入`$$\frac{P_i}{\sum_{j \in \mathbf{V}}^NP_j}$$`, 输完之后 Enter， 你看到了什么？ 

$$\frac{P_i}{\sum_{j \in \mathbf{V}}^NP_j}$$

这个时候会有同学说，可是这些符号，我怎么记得住呢？
我给大家提供了一个参考手册，大家有空就看看 https://github.com/Artificial-Intelligence-for-NLP/comment-setimental-classification/blob/master/Latex-Symbols.pdf ，熟能生巧。 

```


## Mr.Gao的lesson目录
```md
第一部分 基础篇 经典人工智能模型方法 
第一周 人工智能导论与语法树，自动机理论
 1.1 形式语言与语法树 
1.1.1）形式语言与正则表达式
1.1.2) 语法树与语言表示
 1.1.3) 使用语法树自动生产语言
 1.2 自动机理论 
 1.2.1) 有限自动机基础
 1.2.2) 语法树解析
 1.2.3) 语法树解析的实际例子
 1.3 作业：基于Syntax Tree实现西部世界对话智能系统 
 1.3.1) 数据驱动的编程
1.3.2) Python 实现句子生成与语法解析
第二周 智能搜索策略
 2.1 智能搜索方法论 
 2.1.1）搜索问题与决策问题
 2.1.2) 智能搜索典型问题分析(传教士过河、八皇后等问题)
 2.2 智能搜索的实现 
 2.2.1）深度搜索，广度搜索与最优搜索
 2.2.2）搜索剪枝问题
2.3 作业：北京市地铁自动换乘 
2.3.1）数据获取爬虫的建立
2.3.2) 编写智能搜索Agent
 
第三周 动态规划与线性优化：
 3.1)优化问题 
 3.1.1) 优化问题的背景
 3.1.2) 优化问题的现状与常用方法
 3.2)动态规划 
 3.2.1) 动态规划的原理
 3.2.2) 动态规划的典型实例
 3.2.3) Python 实现动态规划的最佳实践
 3.3)线性优化 
 3.3.1) 线性优化的原理
 3.3.2) 线性优化经典作业
 3.3.2) Python 线性优化最近实践
3.4)作业：上海市外卖小哥送餐路线规划问题 
3.4.1) 问题复杂度分析
3.4.2) 使用动态规划解决实例
3.4.3) 使用线性规划解决实例
第四周 自然语言理解初步
 4.1 词向量 
4.1.1)文本表示初步
 4.1.2)降维与 embedding 的原理；
 4.1.3)词向量初步知识
 4.1.4)Python 词向量使用的最佳实践
4.2 关键词提取 
4.2.1）关键字提取的主要方法与挑战
4.2.2) 基于频率的TFIDF
4.2.3) 基于图关系的 Text-Rank
4.2.4) 基于机器学习的方法
4.2.5) 基于词向量与图网络的方法
4.2.6) Python 关键词提取的最近实践
4.3 实体识别 
4.3.1) 实体识别的原理与现状
4.3.1) 实体识别的应用场景
4.3.2) Python 实体识别的最佳实践
4.4 依存分析 
4.4.1) 依存分析的原理与现状
4.4.2) Python 依存分析的最佳实践
 
第五周：搜索引擎与文档检索：
5.1 自动检索系统 
5.1.1) 搜索引擎与文档搜索的背景
5.1.2) 基于关键字的文本搜索
5.1.3) 布尔代数搜索
5.2 PageRank 
5.2.1) PageRank 原理
5.2.2) PageRank 的其他应用场景
5.2.3) Python 实现大规模搜索系统的关键能力与算法实例
项目实训一：新闻人物言论自动提取 或 PDF 重点信息智能标准系统
 - 数据获取，数据转换，数据标准化
- 词向量的构建
- 依存分析与实体识别，重要信息识别
 - 搜索系统
 - 综合实现
 
第二部分 机器学习与深度学习
第六周 统计概率模型：
 6.1 语言模型 
6.1.1) 语言模型的历史背景与意义
 6.1.2) 语言模型的种类
 6.2 统计概率模型 
 6.2.1) 条件概率
 6.2.2) Python统计语言概率的实现
 6.2.3) Good-Turing Estimation 
 6.3 编辑距离与文本相似度 
6.3.1)文本相似度的主要方法
 6.3.2)编辑距离的原理
 6.3.3)编辑距离的 python 实现
6.4 作业：中文拼写错误自动纠正 
6.4.1 语言模型的构建
6.4.2 编辑距离的计算
6.4.3 自动纠错算法的实现
6.4.4 Python源代码完整分析
第七周 经典机器学习一
7.1 机器学习的历史发展与原理 
7.1.1) 机器学习的背景与原理
7.1.1) 机器学习的主要流派
7.1.2) 机器学习的现状分析
7.2 过拟合与欠拟合 
7.2.1) Bias和 Variance
7.2.2) 模型能力的分析
7.2.3) 数据能力的分析
7.2.4) 过拟合与欠拟合的原理与策略
7.3 训练集，测试集，准确度 
7.3.1) 数据对机器学习模型的影响 
7.3.2) 训练集、测试集、准确度的关系 
第八周：经典机器学习二：
 8.1 经典机器学习模型 
 8.1.1）回归和分类
 8.1.2）逻辑回归
8.1.3）贝叶斯分类器
8.1.4）KNN模型，
 8.1.5）SVM
8.1.6）决策树
8.1.7 Python）机器学习模型的最近实践
8.2 Ensemble 机器学习方法 
8.2.1）Ensemble 机器学习的原理
8.2.2）Random Forest 随机森林
 8.2.3）XGBOOST模型
第九周：经典机器学习三：
 9.1 非监督/半监督学习与聚类模型： 
9.1.1）K-Means算法与实例
 9.1.2）层次聚类与实例
 9.1.3）基于 embedding 的聚类机器实例
9.2 机器学习常见实践问题分析 
9.2.1）天气预测
9.2.2）文本分类
9.2.3）图像分类
9.2.4）机器阅读理解
9.2.5）博弈问题
9.3 作业：实现贝叶斯分类器，依据药物说明书进行药物适应症自动识
别 
第十周 深度学习初步
10.1 神经网络 
10.1.1) Loss函数，Backpropagation
10.1.2) 梯度下降
10.1.3) softmax, cossentropy
10.1.4) Optimizer 优化器
10.2 神经网络的实践分析 
10.2.1) 模型的稳定性 
10.2.2) 模型的可解释性 
10.2.3) 模型的运行分析 
10.5 作业：手动从零实现一个神经网络模型 
10.3.1) 实现神经元
10.3.2) 实现拓扑排序
10.3.3) 实现 Backpropagation
10.3.4) 实现神经元权重自动调整
10.3.5) 利用完成的神经网络模型进行真实机器学习任务
项目作业二：细粒度客户评论自动分类
 
 - 数据预处理过程
 - 数据分析与整理
 - 模型的分析与搭建
 - 模型的调整与分析
 - 模型的部署与发布
第十一周 word2vec 
11.1 word embedding与词向量 
11.1.1) 词向量的原理
11.1.2) 哈夫曼树与 Negative Samples
11.1.3) GloVe, CoVe, ELMO 等高级词向量方式
11.1.4) Python 利用神经网络训练词向量的实例
11.2 句子向量 
11.2.1) 句子向量的使用场景与背景
11.2.2) 句子向量的构建与评价标准
11.2.3）Python 构建句子向量的实例
11.3 词向量的高级用法 
11.3.1)利用词向量找到隐藏重要词汇
11.3.2)利用词向量找到新词汇
11.4 作业：使用词向量自动整理同义词 
 
第十二周 CNN卷积神经网络
 12.1 卷积神经网络与 Spatial Invariant 
 12.1.1) 卷积神经网络的历史背景
 12.1.2) 卷积神经网络空间平移不变形(Spatial Invariant)的原理
12.1.3) 卷积神经网络与 weights sharing
 12.1.4) 卷积神经网络的原理及Python 实现
 12.2 Pooling, Dropout 与 Batch Normalization 
 12.2.1) Pooling
 12.2.2) Dropout
 12.2.3) Batch Normalization
 12.3 CNN 的可视化 
 12.4 经典 CNN 模型分析： 
 12.4.1) LeNet
12.4.2) AlexNet
 12.4.3) GoogLeNet
 12.4.4) VGG, ResNet
 12.4.5) DenseNet
12.5 作业：进行萝莉和正太的分类 
12.5.1) Python 深度学习环境的搭建
12.5.2) Keras, Tensorflow 的介绍与使用方法
12.5.2) 使用 Keras 搭建CNN 模型模型
12.5.3) 模型的调试与优化
12.5.4) 模型的发布
第十三周 RNN循环神经网络
13.1)序列模型 
13.1.1) 时间序列问题的分析
13.1.2) 时间序列模型存在的问题挑战
13.2)RNN 循环模型 
13.2.1) RNN 的原理
13.2.2) RNN 的相关问题
13.3)LSTM 与 GRU 
13.3.1) LSTM的原理与实现
13.3.2) GRU 的原理
13.3.3) Python 进行 RNN 模型的最佳实践
13.4)RNN 训练的高级问题 
13.5)Transfer Learning 迁移学习 
 13.5.1) 迁移学习的背景
 13.5.2) 迁移学习的方法
 13.5.3) Python 实现迁移学习的最佳实践
 
第十四周 深度学习高级问题
14.1 Seq2Seq, Transform, BERT 
14.1.1) Seq2Seq 的原理
14.1.2) Seq2Seq 中的搜索方法
14.1.3) Attention注意力机制
14.1.4) Python Seq2Seqm模型的最佳实践
14.1.4) Transform 机制及其应用
14.1.5) BERT 原理及其应用
14.2 强化学习 
14.2.1) 强化学习的原理
14.2.2) 强化学习的常用方法与实例
14.2.3) 强化学习面临的挑战
14.3 自动对话机器人、文本自动摘要生成、文本自动阅读理解、自动
驾驶等深读学习高级问题 
第十五周：面向服务的智能客户机器人与新闻自动摘要生成
15.1 实训三： 文本自动摘要系统的构建讲解与导引
 - 自动摘要的问题背景与挑战
 - 中文文本摘要遇到的问题
 - 使用 TextRank 进行文本自动摘要的实现
 - 使用 Sentence Embedding 句子向量进行文本自动摘要的实现
 - 完整的文本摘要系统所需要的技术能力分析
15.2 实训四： 面向服务的对话机器人的构建讲解与导引
- 对话机器人的历史背景
- 使用语法树进行对话的实现
- 意图分析与识别
- 文本相似度匹配
- 文本快速检索
- 对话机器人的整体架构分析
第十六周：目前人工智能与局限性、前沿 NLP 问题的现状及发展情况
 16.1 学习能力迁移问题，样本迁移问题 
 16.2 机器学习的可解释性 
 16.3 非结构数据的处理的计算 
 16.4 经典 AI 模型的计算复杂性 
 16.5 AI 产业化面临的问题： 
 16.5.1) 数据标注与结构化数据
 16.5.1) 问题定义与认识
 16.5.2) 少量数据与机器学习的不可行性
 16.6.基于人类背景知识的常识推理与认知问题 
 
```