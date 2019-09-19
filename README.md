# AI&NLP
## 记录学习NLP之路，一起加油 learn NLP together, code for AI

* NLP每个课程的作业和代码放在对应lesson-0X
* 我的规范：code/review过程中，记录经验和笔记，用【我的笔记】做开头，上下用''' '''注释
* 开头默认用模板，详细描述代码使用，方便后面整理和review（设置pycharm自动生成）, 详情见里面的代码

## lesson-01 知识点
- 待补充

## lesson-02 知识点
- 待补充
- 爬虫获取广州地铁路线的数据
- 利用广度/深度优先搜索算法，获取2个地铁站之间的换乘路线
- 因为百度的页面太不规范，导致有几个页面无法获取数据，故我使用的是WIKI百科的，https://zh.wikipedia.org/wiki/%E5%B9%BF%E5%B7%9E%E5%9C%B0%E9%93%81    
- 使用高德地图API获取坐标：https://map.amap.com/subway/index.html，根据坐标点绘制广州地铁图
![image](https://user-images.githubusercontent.com/9695113/65223170-2d188300-daf3-11e9-91a7-0b3a2c7bd077.png)
- 使用的技术：爬虫-beautifulsoup-数据处理, re, bfs, dfs等

## lesson-03 知识点
- 待补充

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

## lesson-05 知识点

- NER
- TFIDF
- wordcloud
- 用jieba 词性——解决80%左右的问题
- jieba, pyltp,coreNLP 三个的比较

- pyltp 中文分词的使用

#### 面试题：为什么要用cos距离，夹角的距离来计算文本的相似度呢？

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


## lesson-07
- 目标：判别一篇新闻的来源是否为新华社
- 重点：样本中有 87% 来源是新华社，因此低于87%的的判别可以认为是无效的
- 方法：TFIDF向量化、Precision； Recall； F1 Score ；ROC AUC score
- 对以下方法均计算了上述参数： Logistic Regression、KNN、SVM、Naive Bayes、Random Tree、 Random Forest
- 最后在87052篇新闻中，找出了180篇疑似抄袭的新闻

## lesson-08知识点 
在做所有机器学习问题时：
Important of Preprocessing
* Balance Analysis, 确定一个基准Base line
* Remove Noise, outliers问题
* Remove Collinearity, 数据越大，纬度越大（纬度灾难）
* Rescale Inputs, lesson-07提到

老师，发一下你之前讲的SVM推导的视频吧
Minchiuan Gao对所有人说： 09:11 PM
https://zoom.us/recording/share/j5O4MP9x5eIBYl-CFbBBSCf8ySTaCy_Zt3BRDIocJRawIumekTziMw
密码： AI@NLP
Xin对所有人说： 09:12 PM
显示密码错误
Minchiuan Gao对所有人说： 09:12 PM
A1@NLP

- 笔记：学习这几个模型，要学会科学的思维
- 怎么调参数呢？——1. 观察    2. 猜想   3. 动手实践    4. 检查与验证

- 面试题：
- 1. 为什么叫SVM？
- 2. 什么叫核函数
- 3. SVM的空间变换
- 4. KNN-速度慢，空间大
- 5. 为什么叫朴素贝叶斯，PPT的计算题
- K-means的时间复杂度是多少？（阿里内部还是有用到K-means的）


## Sci-Computing
- SciPythonTutorial1-Numpy介绍.ipynb
- SciPythonTutorial2-Matplotlib.ipynb
- SciPythonTutorial3-Pandas_介绍.ipynb
- SciPythonTutorial3-Pandas_Cheat_Sheet.pdf
  
  
##### numpy必知必会26问.py
##### Numba介绍.ipynb


## Books
- 数学之美.pdf
- 深度学习-[美]Ian+Goodfellow（伊恩·古德费洛）.pdf

- 《Python自然语言处理实战：核心技术与算法》.pdf
- 机器学习实战：基于Scikit-Learn和TensorFlow.pdf
- 神经网络与深度学习（仅限学习交流请勿传播）-邱锡鹏.pdf
- 斯坦福CS224n_自然语言处理与深度学习_笔记_hankcs.pdf
- word2vec.pdf
- Visualizing-Data-using-t-SNE.pdf


- 流畅的python.pdf
- FlaskWeb开发：基于Python的Web应用开发实战.pdf
- linux简单命令速查表.pdf


- 《百面机器学习算法工程师带你去面试》中文PDF.pdf



## requirements.txt
- 生成指南：
- 第一步：安装包 pip install pipreqs
- 第二步：在对应路径cmd，输入命令生成 requirements.txt文件：pipreqs ./ --encoding=utf8 避免中文路径报错
- 第三步：下载该代码后直接pip install -r requirements.txt

- QA
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