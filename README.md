# AI&NLP
## 记录学习NLP之路，一起加油 learn NLP together, code for AI

* NLP每个课程的作业和代码放在对应lesson-0X
* 我的规范：code/review过程中，记录经验和笔记，用【我的笔记】做开头，上下用''' '''注释
* 开头默认用模板，详细描述代码使用，方便后面整理和review（设置pycharm自动生成）, 详情见里面的代码

## lesson-01 知识点
- 待补充

## lesson-02 知识点
- 待补充

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
- jieba, nlp,coreNLP 三个的比较

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

