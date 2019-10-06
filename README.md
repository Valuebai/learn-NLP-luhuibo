# AI&NLP
## 记录学习NLP之路，一起加油 learn NLP together, code for AI


![Languages](https://img.shields.io/badge/Languages-Python3.6-green)
![Build](https://img.shields.io/badge/Build-passing-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-orange.svg)
![Contributions](https://img.shields.io/badge/Contributions-welcome-ff69b4.svg)


* NLP每个课程的作业和代码放在对应lesson-0X
* 我的规范：code/review过程中，记录经验和笔记，用【我的笔记】做开头，上下用''' '''注释
* 开头默认用模板，详细描述代码使用，方便后面整理和review（设置pycharm自动生成）, 详情见里面的代码

---
## Books

### **入门书籍**
> - 《统计学习方法(李航)》.pdf
>  - 《统计学习方法》可以说是机器学习的入门宝典，许多机器学习培训班、互联网企业的面试、笔试题目，很多都参考这本书。
>  - [github:将李航博士《统计学习方法》一书中所有算法实现](https://github.com/WenDesi/lihang_book_algorithm)


> - 《机器学习》西瓜书_周志华.pdf
>  - 周志华老师的《机器学习》（西瓜书）是机器学习领域的经典入门教材之一，周老师为了使尽可能多的读者通过西瓜书对机器学习有所了解, 所以在书中对部分公式的推导细节没有详述，但是这对那些想深究公式推导细节的读者来说可能“不太友好”，本书旨在对西瓜书里比较难理解的公式加以解析，以及对部分公式补充具体的推导细节
>  - [github:《机器学习》（西瓜书）公式推导解析](https://github.com/datawhalechina/pumpkin-book)


> - 《机器学习实战(Peter Harrington著) 中文》.pdf
>  - 好多人推荐的
>  - 每一章都用代码实现了一部分机器学习的算法，对理解算法的原理很有帮助。
>  - [数据以及参考code可在官网下载](https://www.manning.com/books/machine-learning-in-action)

> - 深度学习的数学.[日]涌井良幸
> - 生动有趣
> - 暂时未有电子版，得购买纸质版本
> - [图灵程序设计丛书：深度学习的数学](https://www.ituring.com.cn/book/2593)


> - 《Python自然语言处理实战：核心技术与算法》.pdf
>  - 这本书用来过一遍用到的NLP方向的技术，里面很多技术未详细说明
>  - [github:读书笔记一](https://github.com/SysuJayce/NLP_learn)
>  - [github:读书笔记二](https://github.com/cunyu1943/python_nlp_practice)


### **必读经典**
> - 《数学之美》吴军.NLP学习必看.pdf

> - 《深度学习》花书_ AI圣经(Deep Learning) .pdf
>  - 由于太厚太难，不适合入门，有基础再看好点，Elon.Mask推荐


### **其他的资料**
> - 《机器学习实战：基于Scikit-Learn和TensorFlow》.pdf
> - 《神经网络与深度学习》-邱锡鹏.pdf
> - ppt-深度学习-李宏毅-宝可梦老师.pptx
>  - bilibili上有视频，这个台湾老师很可爱
> - 斯坦福CS224n_自然语言处理与深度学习_笔记_hankcs.pdf
> - Deep Learning 实战之 word2vec.pdf
> - linux简单命令速查表.pdf

### **算法相关**
> - 刷leetcode：https://leetcode-cn.com/

> - 算法《漫画算法：小灰的算法之旅》.pdf

> - 算法小说《算法神探》一口气就能看完的神奇算法书.pdf


### **面试书&资料**
> - 面试《百面机器学习算法工程师带你去面试》中文PDF.pdf
>   - 收录了超过100道机器学习算法工程师的面试题目和解答，其中大部分源于Hulu（美国著名视频网站）算法研究岗位的真实场景。这本书不仅囊括了机器学习的基本知识，而且还包含了成为优秀算法工程师的相关技能，更重要的是凝聚了笔者对人工智能领域的一颗热忱之心，旨在培养读者发现问题、解决问题、扩展问题的能力，建立对机器学习的热爱，共绘人工智能世界的宏伟蓝图。
> - 面试《美团机器学习实践 》.pdf
>   - 全面介绍了美团在多个重要方面对机器学习的应用，涵盖搜索、推荐、广告、风控、机器学习、计算机视觉、语音、自然语言处理、智能调度、机器人和无人配送等多个技术方向。


### **python相关书籍**
> 这里就不记录了

---
## lesson-01 知识点
- 待补充
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
- 待补充

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

- NER
- TFIDF
- wordcloud
- 用jieba 词性——解决80%左右的问题
- jieba, pyltp,coreNLP 三个的比较

- pyltp 中文分词的使用

**面试题：为什么要用cos距离，夹角的距离来计算文本的相似度呢？**

> Project-01 NewsInfo-Auto-Extration 新闻任务言论自动提取
> - github: https://github.com/Valuebai/NewsInfo-Auto-Extration

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
- 目标：判别一篇新闻的来源是否为新华社
- 重点：样本中有 87% 来源是新华社，因此低于87%的的判别可以认为是无效的
- 方法：TFIDF向量化、Precision； Recall； F1 Score ；ROC AUC score
- 对以下方法均计算了上述参数： Logistic Regression、KNN、SVM、Naive Bayes、Random Tree、 Random Forest
- 最后在87052篇新闻中，找出了180篇疑似抄袭的新闻


---
## lesson-08知识点 

**在做所有机器学习问题时：Important of Preprocessing**

- Balance Analysis, 确定一个基准Base line
- Remove Noise, outliers问题
- Remove Collinearity, 数据越大，纬度越大（纬度灾难）
- Rescale Inputs, lesson-07提到

```
老师，发一下你之前讲的SVM推导的视频吧
Minchiuan Gao对所有人说： 09:11 PM
https://zoom.us/recording/share/j5O4MP9x5eIBYl-CFbBBSCf8ySTaCy_Zt3BRDIocJRawIumekTziMw
密码： AI@NLP
Xin对所有人说： 09:12 PM
显示密码错误
Minchiuan Gao对所有人说： 09:12 PM
A1@NLP
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

> 新华社新闻抄袭自动判别
```
任务要求：

1. 构建一个机器学习模型，判断这个文章是不是新华社的
2. 当这个模型的acc 大于 0.8778， recall， precision，f1等值都较高的时候
3. 用该模型 判断一篇文章是否是新华社的文章，如果判断出来是新华社的，但是，它的source并不是新华社的，那么，我们就说，这个文章是抄袭的新华社的文章
4. Text Representation uses "https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html"
```



---
## lesson-09知识点 

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


---
## lesson-11知识点



---
## lesson-13知识点

**课程笔记**
# 后面再整理进对应的readme.md中
- RNN公式


- LSTM(Long Short Term Memory，一种特殊的RNN)主要解决什么问题？？
- 解决：LSTM专门设计用来避免长期依赖问题的
- 主要用来：用门gate的结构来对单元状态的增加或者删除信息。（门是选择性让信息通过的方式）
PPT中的output gate公式，上课时没搞懂，一串公式看着头大（视频中最核心的点，看回放）

比喻：人与人之间的信息传递：A-B-C-D，用LSTM相当于在中间加了i,j,k，作为中间人

- LSTM的变种（变体）形式：GRU（Gated Recurrent Unit）
 - 提出的论文地址：http:arxiv.org/pdf/1406.1078v3.pdf
 - GRU将输入门和遗忘门结合成一个单独的“更新门”
 
 
 尤其是Ct = ft * C_t-1 + i_t * ~C_t 的理解

```markdown


LSTM和GRU的应用场景比较：
- LSTM 有Forget Gate（控制C_{t-1}）和 Input Gate（控制新的C_t保留程度）
- GRU : (1-Zt)-> Forgate, Zt -> Input Gate

- GRU模型精简，参数少，拟合能力相对比较弱，适用于小规模不是很复杂的数据集
- LSTM 参数多，拟合能力强，适合大规模复杂度高的数据集

```

- ReLu函数（常用的激活函数）的公式和图像长什么样子？？
```markdown
ReLu函数在SGD中能够快速收敛。
深度学习中最大的问题是梯度消失问题（~下面解释），使得训练网络收敛越来越慢，而ReLu凭借其线性、非饱和的形式，训练速度则快很多。
~：神经网络在进行方向误差传播时，各个层都要乘以激活函数的一阶导数G=e·∅'(x)·x，梯度每传递一层都会衰减一层，网络层数较多时，梯度G就会不停衰减知道消失

公式：
y = 0 (x<0)
y = x (x>0)

图像：

```

- 举个例子说明下：用词向量进行文本分类的例子

### kaggle中使用
作业-project恶意评论分类挑战

- 学会使用kaggle
- kaggle的竞赛经历

视频中用到的是kaggle.com/yekenot/pooled-gru-fasttext
改动kaggle-notebook中的代码，让效果超过上面的，把代码保存下来，为项目3做准备

- 作业中的书和论文，主要来理解RNN

assignment5：transger-learning的论文，主要讲seq2seq，里面有涉及RNN这些的



**面试题：**
1. LSTM如何防止梯度爆炸
2. LSTM视频中的图，要非常熟悉

- 工作中的效果达不到领导的要求——说训练数据不够

---
## Sci-Computing
- SciPythonTutorial1-Numpy介绍.ipynb
- SciPythonTutorial2-Matplotlib.ipynb
- SciPythonTutorial3-Pandas_介绍.ipynb
- SciPythonTutorial3-Pandas_Cheat_Sheet.pdf
  
  
##### numpy必知必会26问.py
##### Numba介绍.ipynb


---
## requirements.txt
- 生成指南：
- 第一步：安装包 pip install pipreqs
- 第二步：在对应路径cmd，输入命令生成 requirements.txt文件：pipreqs ./ --encoding=utf8 避免中文路径报错
- 第三步：下载该代码后直接pip install -r requirements.txt

## 检查python的语法正确性
- 第一步：安装包 pip install --upgrade pyflakes
- 第二步：用这个命令（pyflakes xxx.py）就能检查xxx.py文件中的错误。

---
## 《统计学习方法》一书
**这本书的第一章：统计学习方法概论，里面有很多内容解释对理解和学习机器学习很有帮助，需要找个时间精简下做成笔记放在这里**

---
## QA
问题：主题模型和lda和拉普拉斯变换

答案：

问题：对于0/1分类变量/数值变量除了归一化还有其他处理法吗

答案：

问题：前后文关系和句法分析和知识图谱

答案：

问题：切词切不好和遇到新词以及源代码改进

答案：

问题：两堆名字，对应论文和著作文，求最快匹配

答案：

问题：特征工程：特征转换，特征提取

答案：