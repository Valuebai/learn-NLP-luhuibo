
## Project-02-文本摘要，项目2的开始
```markdown

## Auto Summarization
#### Main References: 
1. TextRank: https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf
2. SentenceEmbedding: https://openreview.net/pdf?id=SyK00v5xx
3. InformationRetrievalAndTextMining: https://github.com/Artificial-Intelligence-for-NLP-Chinese/References/tree/master/information-retrieve

——from Mr.Gao[https://github.com/Computing-Intelligence/AutoSummarization]
```

## 目前的摘要技术分为
### 1. Extraction 抽取式
### 2. Abstraction 生成式

> 目前Extraction抽取式的主要方法：

>> - 基于统计：统计词频，位置等信息，计算句子权值，再简选取权值高的句子作为文摘，特点：简单易用，但对词句的使用大多仅停留在表面信息。

>> - 基于图模型：构建拓扑结构图，对词句进行排序。例如，TextRank/LexRank。

>> - 基于潜在语义：使用主题模型，挖掘词句隐藏信息。例如，采用LDA，HMM。

>> - 基于线路规划：将摘要问题转为线路规划，求全局最优解。

>> 在python语言中用于文本摘要自动提取的库包括goose，SnowNLP，TextTeaser，sumy，TextRank等。


## PageRank算法简介：

![图 1 PageRank算法](https://www.github.com/Valuebai/my-markdown-img/raw/master/小书匠/1569569319714.png)


假设我们有4个网页——w1，w2，w3，w4。这些页面包含指向彼此的链接。有些页面可能没有链接，这些页面被称为悬空页面。

![enter description here](https://www.github.com/Valuebai/my-markdown-img/raw/master/小书匠/1569569411160.png)

- w1有指向w2、w4的链接

- w2有指向w3和w1的链接

- w4仅指向w1

- w3没有指向的链接，因此为悬空页面


为了对这些页面进行排名，我们必须计算一个称为PageRank的分数。这个分数是用户访问该页面的概率。

为了获得用户从一个页面跳转到另一个页面的概率，我们将创建一个正方形矩阵M，它有n行和n列，其中n是网页的数量。

![enter description here](https://www.github.com/Valuebai/my-markdown-img/raw/master/小书匠/1569569472985.png)


矩阵中得每个元素表示从一个页面链接进另一个页面的可能性。比如，如下高亮的方格包含的是从w1跳转到w2的概率。

![enter description here](https://www.github.com/Valuebai/my-markdown-img/raw/master/小书匠/1569569512404.png)

**如下是概率初始化的步骤：**

1. 从页面i连接到页面j的概率，也就是M[i][j]，初始化为1/页面i的出链接总数wi 

2. 如果页面i没有到页面j的链接，那么M[i][j]初始化为0

3. 如果一个页面是悬空页面，那么假设它链接到其他页面的概率为等可能的，因此M[i][j]初始化为1/页面总数

因此在本例中，矩阵M初始化后如下：
![enter description here](https://www.github.com/Valuebai/my-markdown-img/raw/master/小书匠/1569569625983.png)

最后，这个矩阵中的值将以迭代的方式更新，以获得网页排名。



## TextRank自动摘要算法简介

**现在我们已经掌握了PageRank，让我们理解TextRank算法。我列举了以下两种算法的相似之处：**

- 用句子代替网页

- 任意两个句子的相似性等价于网页转换概率

- 相似性得分存储在一个方形矩阵中，类似于PageRank的矩阵M


TextRank算法是一种抽取式的无监督的文本摘要方法。让我们看一下我们将遵循的TextRank算法的流程：

![enter description here](https://www.github.com/Valuebai/my-markdown-img/raw/master/小书匠/1569570219876.png)


1. 第一步是把所有文章整合成文本数据

2. 接下来把文本分割成单个句子

3. 然后，我们将为每个句子找到向量表示（词向量）。

4. 计算句子向量间的相似性并存放在矩阵中

5. 然后将相似矩阵转换为以句子为节点、相似性得分为边的图结构，用于句子TextRank计算。

6. 最后，一定数量的排名最高的句子构成最后的摘要。


## TextRank用于关键词提取的算法如下：


1. 把给定的文本T按照完整句子进行分割，即 T = [S1, S2, S3, ... ,Sm]

2. 对于每个句子Si属于T，进行分词和词性标注处理，并过滤掉停用词，只保留指定词性的单词，如名词、动词、形容词，即Si = [ti,1 , ti,2 , ... , ti,n],其中 ti,j 是保留后的候选关键词。

3. 构建候选关键词图G = (V,E)，其中V为节点集，由（2）生成的候选关键词组成，然后采用共现关系（co-occurrence）构造任两点之间的边，两个节点之间存在边仅当它们对应的词汇在长度为K的窗口中共现，K表示窗口大小，即最多共现K个单词。

4. 根据上面公式，迭代传播各节点的权重，直至收敛。

5. 对节点权重进行倒序排序，从而得到最重要的T个单词，作为候选关键词。

6. 由5得到最重要的T个单词，在原始文本中进行标记，若形成相邻词组，则组合成多词关键词。


---
## 参考资料
> [手把手 | 基于TextRank算法的文本摘要（附Python代码）](https://mp.weixin.qq.com/s?__biz=MjM5MTQzNzU2NA==&mid=2651666276&idx=3&sn=fbd030dda4318ca328fcf918fc1df1e3&chksm=bd4c10f78a3b99e1b6ebfe362a969f36f39e72ce93336bf3fe4fa03b237120156a82c8becfd4&mpshare=1&scene=1&srcid=0926aQxlE7uoNlUkHMjmqwpV&sharer_sharetime=1569463898414&sharer_shareid=7429dfbe8eeefb16193f3b889173524e&key=1d1ca7b6df234b6eb8721eb961bac936b31eea52d6df0f9da7f23ed3ac6a644973c0ce22b6d73db01420f94551b51ba4ff3373167795ed451f3c89779f5c0ab3936d1403bee433d2c0815daaab647ab5&ascene=1&uin=MTE2NTM1NQ%3D%3D&devicetype=Windows+10&version=62070141&lang=zh_CN&pass_ticket=FjR0fPMrSxrRHa7VbnHhBJ%2BAFLgOTXEKsBcxoKN7PN8%3D)
>> 里面的过程可以参考下，有助于理解文本自动摘要的过程














---
## 课堂的笔记
- word2vec: FastText
- 句子向量：2019年普林斯顿大学的文章，老师的参考论文
- 文章向量
for sentence in sentences_of_document:
    猪ta = distances(sentence, doc)

<font color=#00ffff size=5>神经网络与词向量框架-Word2Vec的结合（里面有用到神经网络，size=100-300）</font>

> <font color=#00ffff size=5>**知识点回顾**：</font>
>> - CBOW 是输入周围的单词，预测中间的单词——Continue Bag of Words
>> - Skip-Ngram 根据中间的，预测周围的单词
>> - softmax

---

- H~ softmax
  - 有层级的softmax

- Huffman Tree
  - 特点，出现频率越高，就离根节点越近
  - 面试题：
  1. 哈弗曼树要怎么构建的呢？？？
  2. 哈弗曼树在词向量中有什么用？
  3. 怎么解决在反向传播的问题
  
- Negative sampling 负采样
  - PPT的图
  - 不太理解 

早期4个比较有名的词向量方法
- CBOW
- Skip-gram

- FastText
  - （新的词向量方式，类似CBOW,skip-gram）
  - 2017年fb提出，在skip-gram上改进提出
  - 应用：阿里巴巴和蚂蚁金服集团
  - 解决sub-word,out of vocabulary问题，大厂里面经常用到FastText
  
- Glove
  - 产生于斯坦福大学
  - 火过一段时间后，现在没怎么用了

后面seq2seq涉及的3个（学完这门课，需要掌握的7个词向量方法）
- cove
- ELMo
- BERT

---
## 公式的练习

$x_1$

$$数学公式$$

$x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}$
\\(x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}\\)