#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：LuckyHuibo
@Date   ：2019/8/1 16:28
@Desc   ：
将上面经过提取，繁简体转换，结巴分词后的处理txt进行word2vec训练
=================================================='''
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

if __name__ == "__main__":
    infile = './data/wiki-jieba-zh-words.txt'
    outp1 = './data/wiki-zh-model'
    outp2 = './data/wiki-zh-vector'

    model = Word2Vec(LineSentence(infile), size=400, window=5, min_count=5, workers=multiprocessing.cpu_count())
    model.save(outp1)
    model.save_word2vec_format(outp2, binary=False)
