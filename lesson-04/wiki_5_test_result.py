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

设置词云以下面的图片为背景
# mask=hand,  # 背景图片
1.from PIL import Image
2.hand = np.array(Image.open('hang1.jpg'))
# 词语以图片形状为背景分布

=================================================='''

import gensim
import logging
import numpy as np
import matplotlib.pyplot as plt

from gensim import models

from wordcloud import WordCloud
from wordcloud import STOPWORDS
from PIL import Image


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
    # 设置停用词
    stopword = {"请", "有人"}
    # 打开一张图片，词语以图片形状为背景分布
    hand = np.array(Image.open('./data/test.jpg'))
    wc = WordCloud(
        # wordcloud参数配置
        width=1024,
        height=768,
        background_color="white",
        mask=hand,  # 背景图片1.from PIL import Image 2.hand = np.array(Image.open('hang1.jpg')) # 词语以图片形状为背景分布
        # mask=get_mask()
        max_words=300,  # 最大显示的字数
        stopwords=stopword,  # 停用词
        font_path=font,  # 设置中文字体，若是有中文的话，这句代码必须添加，不然会出现方框，不出现汉字
        max_font_size=100,  # 字体最大值
        # random_state=3,  # 设置有多少种随机生成状态，即有多少种配色方案
    )
    wc.generate_from_frequencies(word_cloud)
    # 隐藏x轴和y轴
    plt.axis("off")
    # wc.to_file('wordcloud.png')  # 保存，只保存1张的
    plt.imshow(wc, interpolation="bilinear")
    plt.show()


def run_draw(model, word_str):
    '''
    测试绘制的词云
    word_str：输入词，找这个词的相似词
    model：传入加载好的模型
    :return:
    '''
    logging.basicConfig(format="%(asctime)s:%(levelname)s:%(message)s", level=logging.ERROR)
    # 输入一个词找出相似的前100个词
    result = model.wv.most_similar(word_str, topn=100)
    # 将返回的结果转换为字典,便于绘制词云
    word_cloud = dict()
    for sim in result:
        # print(sim[0],":",sim[1])
        word_cloud[sim[0]] = sim[1]
    # 绘制词云
    draw_word_cloud(word_cloud)


if __name__ == "__main__":
    # # 对wiki训练后的wordvec进行测试
    # model = Word2Vec.load('./data/wiki-zh-model')
    # model2 = Word2Vec.load_word2vec_format('./data/wiki-zh-vector', binary=False)
    # 对bible训练后的wordvec进行测试
    model = models.Word2Vec.load('./data/bible-model-stopwords')
    model2 = gensim.models.KeyedVectors.load_word2vec_format('./data/bible-vector-stopwords', binary=False)

    res1 = model.most_similar('神')
    print(type(res1))
    print(res1)
    res2 = model.most_similar('耶稣')
    print(res2)
    res3 = model.most_similar('祷告')
    print(res3)
    run_draw(model, "神")
    run_draw(model, "耶稣")
    run_draw(model, "大卫")
