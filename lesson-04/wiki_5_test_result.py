#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：LuckyHuibo
@Date   ：2019/8/1 16:35
@Desc   ：测试效果
=================================================='''
from gensim.models import Word2Vec

if __name__ == "__main__":
    model = Word2Vec.load('./data/wiki-zh-model')
    model2 = Word2Vec.load_word2vec_format('./data/wiki-zh-vector', binary=False)

    res1 = model.most_similar('时间')
    print(res1)
    res2 = model.most_similar('广州')
    print(res2)
