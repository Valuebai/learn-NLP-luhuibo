# lesson-13的记录

## 助于理解RNN的优秀文章
> [完全图解RNN、RNN变体、Seq2Seq、Attention机制](https://zhuanlan.zhihu.com/p/28054589)

> [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)


## 作业-project恶意评论分类挑战
Toxic Comment Classification Challeng 以前的kaggle比赛，很值得学习参考，特别是做情感细颗粒的

> https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/notebooks

上面有好多大佬开源分享自己的比赛经验，有太多太多学习的东西，赞

- [一开始啥都不懂的，看下大佬写的For Beginners](https://www.kaggle.com/sbongo/for-beginners-tackling-toxic-using-keras)

---
### 1. 数据EDA
- 不管哪个比赛，对数据的处理往往是特别重要的！
- 下面的链接是大神公开这个比赛的数据EDA+特征工程，必看
> https://www.kaggle.com/jagangupta/stop-the-s-toxic-comments-eda


### 2.Baseline1 - Keras - Bidirectional LSTM baseline ( lb 0.069)
- 使用这个baseline作为作业的第一个模型
- 我的代码命名为：**Keras-Bi-LSTM-Baseline.py**
- 训练结果： loss: 0.0452 - acc: 0.9833 - val_loss: 0.0481 - val_acc: 0.9824, , saving model to ./output/baseline1_weights_base.best.hdf5

- 下面的链接是大神公开的
> https://www.kaggle.com/CVxTz/keras-bidirectional-lstm-baseline-lb-0-069

### 3.Baseline2 - Improved LSTM baseline: GloVe + dropout

- 基于baseline1进行优化改进，训练第二个模型
  - 主要进行2个点的优化
  - glove词向量
  - dropout
- 我的代码命名为：**Baseline2_LSTM_baseline_GloVe_dropout.py**
- 训练结果：519s 4ms/step - loss: 0.0451 - acc: 0.9830 - val_loss: 0.0465 - val_acc: 0.9829
  - output: baseline2_submission.csv

- 下面的链接是大神公开的
> https://www.kaggle.com/jhoward/improved-lstm-baseline-glove-dropout


- 在kaggle比赛中，常常看到glove.6B的词向量，常见的有50d，100d，200d，300d常用英文单词的词向量
> 可以从https://nlp.stanford.edu/projects/glove/上下载
> 如果下载比较慢，可以从这下载，链接：https://pan.baidu.com/s/1m5zKaJGFwV1VNTsHgGqwRw，提取码：5knd

### 4.Baseline3 - NB-SVM strong linear baseline
- 使用NB + SVM作为作业的第三个模型
- 我的代码命名为：**Baseline3_NB-SVM_strong_linear.py**
- 训练结果：直接保存到baseline3_submission.csv
- NB-SVM的理论：NB-SVM的论文支撑.pdf
- 下面的链接是大神公开的
> https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline


### 5.模型结果融合


- **【技巧】在比赛中，经常使用不同模型进行训练，再将结果进行融合看下能否提分**
- 将上面的Baseline2 + Baseline3 的结果进行融合
- 我的代码命名为：**Baseline_ensemble.py**

- 下面的链接是大神公开的
> https://www.kaggle.com/jhoward/minimal-lstm-nb-svm-baseline-ensemble


P.S. input和output文件夹数据太大，故没有上传



## **课程笔记**
- 课堂笔记，详情见：lesson-13下的readme
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

```md
LSTM和GRU的应用场景比较：
- LSTM 有Forget Gate（控制C_{t-1}）和 Input Gate（控制新的C_t保留程度）
- GRU : (1-Zt)-> Forgate, Zt -> Input Gate

- GRU模型精简，参数少，拟合能力相对比较弱，适用于小规模不是很复杂的数据集
- LSTM 参数多，拟合能力强，适合大规模复杂度高的数据集

```

- ReLu函数（常用的激活函数）的公式和图像长什么样子？？
```md
ReLu函数在SGD中能够快速收敛。
深度学习中最大的问题是梯度消失问题（~下面解释），使得训练网络收敛越来越慢，而ReLu凭借其线性、非饱和的形式，训练速度则快很多。
~：神经网络在进行方向误差传播时，各个层都要乘以激活函数的一阶导数G=e·∅'(x)·x，梯度每传递一层都会衰减一层，网络层数较多时，梯度G就会不停衰减知道消失

公式：
y = 0 (x<0)
y = x (x>0)

图像：

```

- 举个例子说明下：用词向量进行文本分类的例子



## kaggle的使用 + 课上笔记
![pic_Toxic Comment Classification Challeng](https://storage.googleapis.com/kaggle-media/competitions/jigsaw/003-avatar.png)
作业-[project恶意评论分类挑战](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge#evaluation)

- 学会使用kaggle
- kaggle的竞赛经历

视频中用到的是[Pooled GRU + FastText score:0.98](https://www.kaggle.com/yekenot/pooled-gru-fasttext)

改动kaggle-notebook中的代码，让效果超过上面的，把代码保存下来，为项目3做准备

- 作业中的书和论文，主要来理解RNN

assignment5：transger-learning的论文，主要讲seq2seq，里面有涉及RNN这些的


## fastText
```md
英文词向量：使用fastText预训练的词向量
http://fasttext.cc/docs/en/english-vectors.html

上面的页面收集了几个使用fastText训练过的词向量。

通过不同来源学习的预训练的词向量有以下几个：

wiki-news-300d-1M.vec.zip: 在维基百科2017、UMBC webbase语料库和statmt.org新闻数据集中(16B tokens)训练的100万个词向量。
wiki-news-300d-1M-subword.vec.zip: 在维基百科2017、UMBC webbase语料库和statmt.org新闻数据集(16B tokens)中包含子单词信息训练的100万个词向量。
crawl-300d-2M.vec.zip: 在Common Crawl训练的200万个词向量 (600B tokens)。

Format
文件的第一行包含词汇表中的单词数量和向量的大小。每一行都包含一个单词及其向量，如默认的fastText格式。每个值都是分开的。单词是按词频降序排列的。这些文本模型可以使用以下代码在Python中加载:

import io

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data

```


