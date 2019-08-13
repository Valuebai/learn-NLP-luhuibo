#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：LuckyHuibo
@Date   ：2019/8/1 16:28
@Desc   ：
将上面经过提取，繁简体转换，结巴分词后的处理txt进行word2vec训练

这里有一些训练词向量的调参技巧：

1.选择的训练word2vec的语料要和要使用词向量的任务相似，并且越大越好，论文中实验说明语料比训练词向量的模型更加的重要，所以要尽量收集大的且与任务相关的语料来训练词向量；
2.语料小（小于一亿词，约 500MB 的文本文件）的时候用 Skip-gram 模型，语料大的时候用 CBOW 模型；
3.设置迭代次数为三五十次，维度至少选 50，常见的词向量的维度为256、512以及处理非常大的词表的时候的1024维；
=================================================='''
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

if __name__ == "__main__":
    # 训练wiki词向量
    infile = './data/wiki-jieba-zh-words.txt'
    outp1 = './data/wiki-zh-model'
    outp2 = './data/wiki-zh-vector'
    # 训练bible词向量
    infile = './data/bible-jieba.txt'
    outp1 = './data/bible-model-stopwords'
    outp2 = './data/bible-vector-stopwords'

    '''
       LineSentence(inp)：格式简单：一句话=一行; 单词已经过预处理并被空格分隔。
       size：是每个词的向量维度； 一般取100-300，或者100-500
       window：是词向量训练时的上下文扫描窗口大小，窗口为5就是考虑前5个词和后5个词； 
       min-count：设置最低频率，默认是5，如果一个词语在文档中出现的次数小于5，那么就会丢弃； 
       workers：是训练的进程数（需要更精准的解释，请指正），默认是当前运行机器的处理器核数。这些参数先记住就可以了。
       sg ({0, 1}, optional) – 模型的训练算法: 1: skip-gram; 0: CBOW
       alpha (float, optional) – 初始学习率
       iter (int, optional) – 迭代次数，默认为5
    '''
    model = Word2Vec(LineSentence(infile), size=400, window=5, min_count=2, workers=multiprocessing.cpu_count())
    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)
