#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：LuckyHuibo
@Date   ：2019/8/28 13:05
@Desc   ：
=================================================='''
import sys
import os

current_dir = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(current_dir)[0]
sys.path.append(rootPath)
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim import models
# 从config配置中读取path_news_txt（保存读取的news_chinese表的数据）, path_news_model（保存的model的路径）文件路径
from config.file_path import path_news_txt, path_news_model, LTP_DATA_DIR

if __name__ == "__main__":
    print('test', path_news_model)
    print('test', LTP_DATA_DIR)
    # 对读取的数据库news进行训练
    news_vec = Word2Vec(LineSentence(path_news_txt), size=100, min_count=1, workers=8)
    # 将训练结果保存为model
    news_vec.save(path_news_model)

    # 加载news_model，进行数据的测试
    model = models.Word2Vec.load(path_news_model)
    # 查找model中跟“说”相关的词
    said = model.most_similar('说')

    '''执行后报错，说训练的model中没有“说”这个词，但是数据库中有【说】字，且min_count=1了

    File "C:\Python36\lib\site-packages\gensim\models\keyedvectors.py", line 464, in word_vec
    raise KeyError("word '%s' not in vocabulary" % word)
    KeyError: "word '说' not in vocabulary"  

    '''

    # 待解决的问题，训练出来的model不包含的【说】KeyError: "word '说' not in vocabulary"
