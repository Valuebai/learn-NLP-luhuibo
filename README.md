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
## lesson-01 知识点
- 待补充
- 人工智能的历史、发展、应用的概述

---
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
## lesson-05 知识点

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

#### 主要讲【机器学习】相关的基础知识

- 机器学习的评价标准

#### 面试题A：为什么叫朴素贝叶斯？？
#### 面试题B：什么是欠拟合，什么是过拟合？在项目中有没有遇到过，是怎么解决的呢？
欠拟合 - 因为模型过于太简单了
过拟合 - 用数据训练出来的A，在实际应用B上不准或不对
作业：总结下overfitting（欠拟合）和 underfitting（过拟合）的原因
#### 面试题C：什么是Bias（偏差）和 Vaiance（方差）？，统计学背景的面试官可能会问


Links：
总结：Bias(偏差)，Error(误差)，Variance(方差)及CV(交叉验证)，https://www.jianshu.com/p/8d01ac406b40


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
## Sci-Computing
- SciPythonTutorial1-Numpy介绍.ipynb
- SciPythonTutorial2-Matplotlib.ipynb
- SciPythonTutorial3-Pandas_介绍.ipynb
- SciPythonTutorial3-Pandas_Cheat_Sheet.pdf
  
  
##### numpy必知必会26问.py
##### Numba介绍.ipynb

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
## requirements.txt
- 生成指南：
- 第一步：安装包 pip install pipreqs
- 第二步：在对应路径cmd，输入命令生成 requirements.txt文件：pipreqs ./ --encoding=utf8 避免中文路径报错
- 第三步：下载该代码后直接pip install -r requirements.txt

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