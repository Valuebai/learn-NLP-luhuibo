## Assignmet: 详情见wiki_1,2,3,4,5开头的，已经main.py

## 利用wiki语料训练模型

详情见wiki_1,2,3,4,5开头的


## 笔记

1. one-hot 表示形式的缺点：
    a. 一般任务词汇量至少1w+，维度灾难
    b. 没有考虑词之间的联系，“词汇鸿沟”
2. 小概念术语： word embedding 和  word2vec。 Word embedding 是 词嵌入，是所有word represent 方法的总称，而word2vec只是其中的一种方式。
3. word2vec 一般常用的两个工具，分别是google的word2vec（gensim），以及 facebook的 fastText，目前中文用的比较多的是 fasttext.  gensim 可以直接pip安装，而fastText需要下载源码安装。
4. word2vec的主要思想：将当前单词与context单词建立联系，当语料足够多时就可以学习语义相似的单词，否则只能学习到用法相似的单词，比如，我爱吃 XXX 口味的 XX。

```md
下面几点是关于word2vec训练的一些注意点：
以fastText中的无监督训练函数为例，详细介绍各个参数的含义：

from fastText import train_unsupervised
model = train_unsupervised(input, model='skipgram', lr=0.05, 
                            dim=100, ws=5, epoch=5, minCount=5, 
                            wordNgrams=1, loss='ns', bucket=2000000, 
                            thread=12, lrUpdateRate=100, t=0.0001, 
                            label='__label__', verbose=2, 
                            pretrainedVectors='')

1.  model： 首先是模型的选择，skip-gram 以及 CBOW。其中skip-gram是给定当前词汇预测上下文单词，而CBOW则是通过上下文预测当前单词。官方给的建议是 skip-gram比较快，但是CBOW虽然比较慢，但是精度会高一点。
2. lr： 学习率的选择需要调试，一般会在0.1左右，太高会报错，太低会训练的比较慢。
3. epoch： epoch 要与 lr 结合考虑，一般不需要超过50次。
4. dim: 得到的向量的维度，语料比较大的话，设置为300，一般语料相应的降低，具体的多大规模设置多少维度，可以尝试50到300，step=50 测试。
5. ws, window size,  表示当前词与预测词在一个句子中的最大距离是多少。context window，比如 ws=5，那么是只考虑前后各两个单词 + 当前单词=5， 还是前后各5个单词+当前=11个单词？ 这个需要看源码确认一下。
6. wordNgrams，默认是使用 1-gram，也就是单独的词，可以尝试 bi-gram（也就是wordNgrams=2），也就是每两个word作为一个unit，会导致词数增加。如果英文的可以考虑，因为英文的每个word就是一个单词，而**中文（建议是1）**的话则是再分好词的基础上进行训练，可以直接设置为1就好，当然可以测试 bi-gram 的情况。
7. loss 默认是 negtive sample， 含义是普通的softmax方法，再输出层每次再保留目标单词的同时，不采用所有的词，而是仅激活其中的一部分（比如100个）单词同 目标单词 作为输出层维度（总词数）。这个可以进行测试，虽然默认的是 ns， 但是看网上demo，用hs的要更多一些，可以进行实验对比。
8. bucket， 基本没有介绍，应该是最多处理多少个词，一般默认就好了。
9. minCount， 这个有时候我们比较关心，就是词频处理，默认是5，也就是词频小于5的词汇我们都不考虑（直接删除掉），这里一般如果语料比较大的话，一般这是为1就好了，不用考虑，词频太低，基本学习不到。

```