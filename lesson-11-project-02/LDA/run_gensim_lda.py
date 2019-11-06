#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：LuckyHuibo
@Date   ：2019/10/21 10:52
@Desc   ：
=================================================='''
from gensim import corpora, models
import jieba.posseg as jp, jieba


# 文本集
texts = [
    '美国教练坦言，没输给中国女排，是输给了郎平' * 99,
    '美国无缘四强，听听主教练的评价' * 99,
    '中国女排晋级世锦赛四强，全面解析主教练郎平的执教艺术' * 99,
    '为什么越来越多的人买MPV，而放弃SUV？跑一趟长途就知道了' * 99,
    '跑了长途才知道，SUV和轿车之间的差距' * 99,
    '家用的轿车买什么好' * 99]
# 分词过滤条件
jieba.add_word('四强', 9, 'n')
flags = ('n', 'nr', 'ns', 'nt', 'eng', 'v', 'd')  # 词性
stopwords = ('没', '就', '知道', '是', '才', '听听', '坦言', '全面', '越来越', '评价', '放弃', '人')  # 停词
# 分词
words_ls = []
for text in texts:
    words = [w.word for w in jp.cut(text) if w.flag in flags and w.word not in stopwords]
    words_ls.append(words)
# 构造词典
print(words_ls)
dictionary = corpora.Dictionary(words_ls)
# 基于词典，使【词】→【稀疏向量】，并将向量放入列表，形成【稀疏向量集】
corpus = [dictionary.doc2bow(words) for words in words_ls]
# lda模型，num_topics设置主题的个数
lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=2)
# 打印所有主题，每个主题显示5个词
for topic in lda.print_topics(num_words=5):
    print('主题:{}'.format(topic))
# 主题推断
print(lda.inference(corpus))
