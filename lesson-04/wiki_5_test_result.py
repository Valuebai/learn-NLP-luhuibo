#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：LuckyHuibo
@Date   ：2019/8/1 16:35
@Desc   ：测试效果
用main.py或者wiki_etrctor.py的词云画一下，效果好点

为什么呢？因为  '大数据'和'共享单车'这两个词不在训练的词向量里面。

为啥不在呢，那么大的维基百科语料库？

1、有可能是我们训练词向量的时候，min_count最小词频默认为5，也就是分词后的语料中词频低于5的词语都不会被训练。
这里可以将词频设置小一点，比如：min_count=2。语料越小，设置的词频应越低，但训练时所消耗的时间也就越长，
大家可以设置试一下。这也是我上面设置min_count=5，只花费17分钟的时间就训练完了。

2、可能语料库里面没有这个词，这么大的语料可能性很小，况且这是最新的维基百科语料。

3、可能繁体字转化为简体字的时候没有转化过来。
=================================================='''
from gensim.models import Word2Vec

if __name__ == "__main__":
    model = Word2Vec.load('./data/wiki-zh-model')
    model2 = Word2Vec.load_word2vec_format('./data/wiki-zh-vector', binary=False)

    res1 = model.most_similar('时间')
    print(res1)
    res2 = model.most_similar('广州')
    print(res2)
