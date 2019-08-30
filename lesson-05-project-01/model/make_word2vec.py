#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：LuckyHuibo
@Date   ：2019/8/28 13:05
@Desc   ：
=================================================='''
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

if __name__ == "__main__":
    news_vec = Word2Vec(LineSentence('../data/news.txt'), size=100, min_count=1, workers=8)
    news_vec.save('../data/news_model')
