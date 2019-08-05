#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：LuckyHuibo
@Date   ：2019/8/5 14:12
@Desc   ：对wiki_extractor_case进行测试
    1、加载训练好的语料corpus
    model = models.Word2Vec.load("./data/zhwiki_news.word2vec")
    #这个zhwiki_news.word2vec是我训练好的，你可以直接Load你训练好的model
    2、wordcloud画图时出现乱码
    ——绘制词云，需要添加.ttf字体，否则会显示乱码的
    .ttf文件在windows里面直接搜索即可，复制到同级目录即可，我用windows搜到的ping.ttf（太大了，未上传）
=================================================='''

# 找出与指定词相似的词
# 返回的结果是一个列表，列表中包含了制定个数的元组，每个元组的键是词，值这个词语指定词的相似度。
import logging
from gensim import models
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def get_mask():
    '''
    获取一个圆形的mask
    '''
    x, y = np.ogrid[:300, :300]
    mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
    mask = 255 * mask.astype(int)
    return mask


def draw_word_cloud(word_cloud):
    '''
    绘制词云，需要添加.ttf字体，否则会显示乱码的
    .ttf文件在windows里面直接搜索即可
    '''
    font = r'./data/ping.ttf'  # 这个太大，需要自己找下
    wc = WordCloud(background_color="white", font_path=font, mask=get_mask())
    wc.generate_from_frequencies(word_cloud)
    # 隐藏x轴和y轴
    plt.axis("off")
    plt.imshow(wc, interpolation="bilinear")
    plt.show()


def test_draw():
    '''
    测试绘制的词云
    :return:
    '''
    logging.basicConfig(format="%(asctime)s:%(levelname)s:%(message)s", level=logging.INFO)
    model = models.Word2Vec.load("./data/zhwiki_news.word2vec")  # 这个zhwiki_news.word2vec是我训练好的，你可以直接Load你训练好的model
    # 输入一个词找出相似的前10个词
    one_corpus = ["心理"]
    result = model.wv.most_similar(one_corpus[0], topn=100)
    # 将返回的结果转换为字典,便于绘制词云
    word_cloud = dict()
    for sim in result:
        # print(sim[0],":",sim[1])
        word_cloud[sim[0]] = sim[1]
    # 绘制词云
    draw_word_cloud(word_cloud)

    # #输入两个词计算相似度
    two_corpus = ["腾讯", "阿里巴巴"]
    res = model.wv.most_similar(two_corpus[0], two_corpus[1])
    print("similarity:", res)

    # 输入三个词类比
    three_corpus = ["北京", "上海", "广州"]
    res = model.wv.most_similar([three_corpus[0], three_corpus[1], three_corpus[2]], topn=100)
    # 将返回的结果转换为字典,便于绘制词云
    word_cloud = dict()
    for sim in res:
        # print(sim[0],":",sim[1])
        word_cloud[sim[0]] = sim[1]
    # 绘制词云
    draw_word_cloud(word_cloud)


if __name__ == "__main__":
    test_draw()
