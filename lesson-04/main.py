#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：LuckyHuibo
@Date   ：2019/7/31 12:01
@Desc   ：主要将wiki,1,2,3,4,5整合进来，方便一起执行
=================================================='''
# langconv这个东西太不好用了，改用hanziconv，直接pip install hanziconv，再from hanziconv import HanziConv
# 使用：HanziConv.toSimplified(要转换的str)
# from libabc.langconv import *  # 开源的繁体转简体库，配合zh_wiki.py使用——直接废除掉！！！
from hanziconv import HanziConv
from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec, Doc2Vec, doc2vec
from gensim.models.word2vec import LineSentence
import jieba
import pandas as pd

# 【官网tutorials】Gensim uses Python’s standard logging module to log various stuff
# at various priority levels; to activate logging (this is optional), run
import logging

# 设置log的格式
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


def preprocess():
    """
    使用gensim中的WikiCorpus库提取wiki的中文语料，并将繁体转成简体中文。
    然后利用jieba的分词工具将转换后的语料分词并写入一个txt
    每个wiki文档的分词结果写在新txt中的一行，词与词之间用空格隔开

    !!!     这个要windows上要跑2个多小时的      !!!

    :return: 对zhwiki...bz2进行提取，并将繁体字转为简体字，存的reduced_zhwiki.txt

    ========================
    from gensim.corpora import WikiCorpus
    import jieba
    from langconv import *       ——这个直接从网上下载后放在同一目录即可
    ========================
    """
    count = 0
    zhwiki_path = './data/zhwiki-20190720-pages-articles-multistream.xml.bz2'
    f = open('./data/reduced_zhwiki.txt', 'w', encoding='utf8')  # 每次成功跑完，最好改下名字，防止重新跑覆盖了
    wiki = WikiCorpus(zhwiki_path, lemmatize=False, dictionary={})
    for text in wiki.get_texts():
        word_list = []
        for sentence in text:
            sentence = HanziConv.toSimplified(sentence)  # 繁体转简体
            seg_list = jieba.cut(sentence)  # 用结巴分词
            for seg in seg_list:
                word_list.append(seg)
        f.write(' '.join(word_list) + '\n')
        count += 1
        if count % 200 == 0:
            print("Saved " + str(count) + ' articles')

    f.close()


def train_w2v():
    """
    用gensim.models.word2vec 对reduced_zhwiki.txt 进行训练

    :return: 数据保存到/data/zhwiki_news.word2vec

========================
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import logging
# 设置log的格式
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)
========================

gensim工具包中详细参数：

在gensim中，word2vec相关的API都在包gensim.models.word2vec中。和算法有关的参数都在类gensim.models.word2vec.Word2Vec中。

算法需要注意的参数有：

 1) sentences: 我们要分析的语料，可以是一个列表，或者从文件中遍历读出。

 2) size: 词向量的维度，默认值是100。这个维度的取值一般与我们的语料的大小相关，如果是不大的语料，比如小于100M的文本语料，则使用默认值一般就可以了。如果是超大的语料，建议增大维度。

 3) window：即词向量上下文最大距离，这个参数在我们的算法原理篇中标记为，window越大，则和某一词较远的词也会产生上下文关系。默认值为5。在实际使用中，可以根据实际的需求来动态调整这个window的大小。如果是小语料则这个值可以设的更小。对于一般的语料这个值推荐在[5,10]之间。

 4) sg: 即我们的word2vec两个模型的选择了。如果是0，则是CBOW模型，是1则是Skip-Gram模型，默认是0即CBOW模型。

 5) hs: 即我们的word2vec两个解法的选择了，如果是0， 则是Negative Sampling，是1的话并且负采样个数negative大于0， 则是Hierarchical Softmax。默认是0即Negative Sampling。

 6) negative:即使用Negative Sampling时负采样的个数，默认是5。推荐在[3,10]之间。这个参数在我们的算法原理篇中标记为neg。

 7) cbow_mean: 仅用于CBOW在做投影的时候，为0，则算法中的为上下文的词向量之和，为1则为上下文的词向量的平均值。在我们的原理篇中，是按照词向量的平均值来描述的。个人比较喜欢用平均值来表示,默认值也是1,不推荐修改默认值。

 8) min_count:需要计算词向量的最小词频。这个值可以去掉一些很生僻的低频词，默认是5。如果是小语料，可以调低这个值。

 9) iter: 随机梯度下降法中迭代的最大次数，默认是5。对于大语料，可以增大这个值。

 10) alpha: 在随机梯度下降法中迭代的初始步长。算法原理篇中标记为，默认是0.025。

 11) min_alpha: 由于算法支持在迭代的过程中逐渐减小步长，min_alpha给出了最小的迭代步长值。随机梯度下降中每轮的迭代步长可以由iter，alpha， min_alpha一起得出。这部分由于不是word2vec算法的核心内容，因此在原理篇我们没有提到。对于大语料，需要对alpha, min_alpha,iter一起调参，来选择合适的三个值。

 12)worker：训练词向量使用时使用的线程数，默认为3。

负采样和层次softmax主要是降低了计算复杂度，提高运算效率

    """
    with open('./data/reduced_zhwiki.txt', 'r', encoding='utf8') as f:
        # 使用gensim的Word2Vec类来生成词向量
        model = Word2Vec(LineSentence(f), sg=0, size=192, window=5,
                         min_count=5, workers=4)  # size默认是100-300，根据你的语料大小进行增加，效果看你的需求
        model.save('./data/zhwiki_news.word2vec')


def test_w2v():
    """
    检测模型训练后的效果，可以根据词的相似度初步进行判断训练的效果
    测试同义词，找几个单词，看下效果
    Q: 我的词向量创建过程有错吗？为啥词向量里没有我想要的词呢？
    A: 这问题我也遇到了 可能你训练的时候词频设置为5  可以把最小词频调低一点
    :return:
    """
    model = Word2Vec.load('./data/zhwiki_news.word2vec')
    # print(model.similarity('大数据', '人工智能'))
    # print(model.similarity('滴滴', '共享单车'))
    print(model.similarity('西红柿', '番茄'))  # 相似度为0.63
    print(model.similarity('西红柿', '香蕉'))  # 相似度为0.44

    word = '中国'
    if word in model.wv.index2word:
        print(model.most_similar(word))

    result = pd.Series(model.most_similar(u'阿里巴巴'))  # 查找近义相关词
    print(result)
    result1 = pd.Series(model.most_similar(u'故宫'))
    print(result1)
    print(model.wv['中国'])  # 查看中国的词向量


class TaggedWikiDocument:
    def __init__(self, wiki):
        self.wiki = wiki
        self.wiki.metadata = True

    def __iter__(self):
        for content, (page_id, title) in self.wiki.get_texts():
            yield doc2vec.LabeledSentence(
                # 1. 对content中的每一个c，
                # 2. 转换成简体中文之后用jieba分词
                # 3. 加入到words列表中
                words=[w for c in content
                       for w in jieba.cut(HanziConv.toSimplified(c))],
                tags=[title])


def train_d2v():
    """
    训练doc2vec
    :return:
    """
    docvec_size = 192
    zhwiki_path = './data/zhwiki-latest-pages-articles.xml.bz2'
    wiki = WikiCorpus(zhwiki_path, lemmatize=False, dictionary={})
    documents = TaggedWikiDocument(wiki)

    model = Doc2Vec(documents, dm=0, dbow_words=1, size=docvec_size,
                    window=8, min_count=19, iter=5, workers=4)
    model.save('./data/zhwiki_news.doc2vec')


if __name__ == '__main__':
    # 第一步，对zhwiki...bz2进行提取，并将繁体字转为简体字，存到reduced_zhwiki.txt
    # 跑一次的时间大概2个多小时
    # 跑完这个要记得注释掉，并修改名称reduced_zhwiki2.txt
    # preprocess()

    # reduced_zhwiki.txt文件太大，打印前5行看下
    # print('=' * 30)
    # f = open('data/reduced_zhwiki.txt', 'r', encoding='utf-8')
    # i = 0
    # for line in f:
    #     print(line)
    #     i += 1
    #     if i == 5:
    #         break
    # f.close()
    # print('=' * 30)

    # 第二步，word2vec
    # train_w2v()

    # 第三步，测试效果
    test_w2v()
