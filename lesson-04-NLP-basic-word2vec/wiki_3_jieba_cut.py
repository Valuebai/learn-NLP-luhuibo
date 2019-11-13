#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：LuckyHuibo
@Date   ：2019/8/1 15:40
@Desc   ：
对处理干净的维基中文语料进行分词，用的是 jieba分词
还有codecs——以前python不支持中文的时候，常用这个，网上查了资料，现在用with oepn和codecs的效果是一样的

由于python中默认的编码是ascii，如果直接使用open方法得到文件对象然后进行文件的读写，都将无法使用包含中文字符
（以及其他非ascii码字符），因此建议使用utf-8编码。


!!!使用word2vec是不需要使用stopwords的!!!

清洗的两条法则
1. 看情况使用stopwords操作，不一定得按照标准的清洗流程
2. 使用于训练的语料尽可能接近预训练的语料，就是未登录词尽量少

=================================================='''
# import codecs   ——因为linix-centOS7无法安装codecs，故使用open即可
import jieba
import logging


def jieba_cut(open_infile, outfile_bible):
    print('分词程序执行开始...')
    i = 0
    with open(outfile_bible, 'w', encoding='utf-8') as f:
        for line in open_infile:
            i += 1
            if i % 1000 == 0:
                print('目前已处理%d个' % i)
            line = line.strip()
            words = jieba.cut(line)
            for word in words:
                f.write(word + ' ')
            f.write('\n')
    print('分词程序执行结束！')


def get_stopwords():
    '''
    加载停用词表，去掉一些噪声
    :return:
    '''
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
    # 加载停用词表
    stopword_set = set()
    with open("stopwords.txt", 'r', encoding="utf-8") as stopwords:  # stopwords.txt停用词表，我放在同一目录下
        for stopword in stopwords:
            stopword_set.add(stopword.strip("\n"))
    return stopword_set


def jieba_cut_with_stopwords(open_infile, outfile_bible):
    # 获取停用词表
    stopwords = get_stopwords()
    print('分词程序执行开始...')
    i = 0
    with open(outfile_bible, 'w', encoding='utf-8') as f:
        for line in open_infile:
            i += 1
            if i % 1000 == 0:
                print('目前已处理%d个' % i)
            line = line.strip()
            words = jieba.cut(line)
            for word in words:
                if word not in stopwords:
                    f.write(word + ' ')
            f.write('\n')
    print('分词程序执行结束！')


if __name__ == "__main__":
    print('主程序执行开始...')
    # 测试bible的词向量训练
    infile_bible = './data/bible-heheben.txt'
    outfile_bible = './data/bible-jieba-stopwords.txt'
    # 测试wiki的语料库训练
    infile = './data/wiki.cn.simple.txt'
    outfile = './data/wiki-jieba-zh-words.txt'

    print('开始读入数据文件...')
    # open_infile = open(infile, 'r', encoding='utf-8')
    open_infile = open(infile_bible, 'r', encoding='utf-8')
    print('读入数据文件结束！')
    ##使用jieba分词，带stopwords的
    # jieba_cut_with_stopwords(open_infile, outfile_bible)
    ##使用jieba分词，不不不带stopwords的
    jieba_cut(open_infile, outfile_bible)
    print('主程序执行结束！')

'''codecs的用法
使用方法
读
下面的代码读取了文件，将每一行的内容组成了一个列表。

import codecs
file = codecs.open('test.txt','r','utf-8')
lines = [line.strip() for line in file]
file.close()
文件读写模式
最为常见的三种模式，见下表，其中模式就是指获取文件对象时传入的参数，最常用的是前三个。
|模式|描述|
|:-:|:-:|
|r|仅读，待打开的文件必须存在|
|w|仅写，若文件已存在，内容将先被清空|
|a|仅写，若文件已存在，内容不会清空|
|r+|读写，待打开的文件必须存在|
|w+|读写，若文件已存在，内容将先被清空|
|a+|读写，若文件已存在，内容不会清空|
|rb|仅读，二进制，待打开的文件必须存在|
|wb|仅写，二进制，若文件已存在，内容将先被清空|
|ab|仅写，二进制，若文件已存在，内容不会清空|
|r+b|读写，二进制，待打开的文件必须存在|
|w+b|读写，二进制，若文件已存在，内容将先被清空|
|a+b|读写，二进制，若文件已存在，内容不会清空|
'''
