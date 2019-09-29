




## kaggle的使用
![pic_Toxic Comment Classification Challeng](https://storage.googleapis.com/kaggle-media/competitions/jigsaw/003-avatar.png)
作业-[project恶意评论分类挑战](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge#evaluation)

- 学会使用kaggle
- kaggle的竞赛经历

视频中用到的是[Pooled GRU + FastText score:0.98](https://www.kaggle.com/yekenot/pooled-gru-fasttext)

改动kaggle-notebook中的代码，让效果超过上面的，把代码保存下来，为项目3做准备

- 作业中的书和论文，主要来理解RNN

assignment5：transger-learning的论文，主要讲seq2seq，里面有涉及RNN这些的

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