#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：LuckyHuibo
@Date   ：2019/7/5 23:42
@Desc   ：
2. 使用新数据源完成语言模型的训练
按照我们上文中定义的prob_2函数，我们更换一个文本数据源，获得新的Language Model:

下载文本数据集（你可以在以下数据集中任选一个，也可以两个都使用）
可选数据集1，保险行业问询对话集： https://github.com/Computing-Intelligence/insuranceqa-corpus-zh/raw/release/corpus/pool/train.txt.gz
可选数据集2：豆瓣评论数据集：https://github.com/Computing-Intelligence/datasource/raw/master/movie_comments.csv
修改代码，获得新的2-gram语言模型
进行文本清洗，获得所有的纯文本
将这些文本进行切词
送入之前定义的语言模型中，判断文本的合理程度
=================================================='''
import pandas as pd
import re
import jieba
from collections import Counter

data_bank_chat_file = 'train.txt'
data_movie_commonent_file = 'train_movie_comments.csv'
data_article_9k = 'train_article_9k.txt'
data_sqlResult = 'train_sqlResult_1558435.csv'


def token(string):
    # 只保留中文、数字、英文
    # 将常用方法写到自己的wiki中：
    # https://github.com/Valuebai/learn-NLP-luhuibo/wiki
    return re.findall('[a-zA-Z0-9\u4e00-\u9fa5]', string)


def jieba_cut(string):
    return list(jieba.cut(string))


def token_1_gram():
    pass


def token_2_gram():
    pass


def prob_1(word):
    return words_count[word] / len(TOKEN)


def prob_2(word_1, word_2):
    if word_1 + word_2 in words_count_2:
        return words_count_2[word_1 + word_2]
    else:
        return 1 / len(TOKEN_2_GRAM)


def prob_sentence(sentence):
    words = jieba_cut(sentence)
    sentence_pro = 1
    for i, word in enumerate(words[:-1]):
        next_ = words[i + 1]
        probability = prob_2(word, next_)
        sentence_pro *= probability
    return sentence_pro


if __name__ == "__main__":
    movie_comment = pd.read_csv(data_movie_commonent_file, encoding='utf-8', low_memory=False)
    movie_comment_list = movie_comment['comment'].tolist()
    print(len(movie_comment_list))
    # 清除列表每一行的多余符号，只保留中文、数字、英文
    clean_comment_list = []
    for line in movie_comment_list:
        clean_comment_list.append(''.join(token(str(line))))
    print('-' * 20 + 'After clean')
    print(clean_comment_list)
    TOKEN = []
    count = 0
    for i in clean_comment_list:
        if count % 10000 == 0:
            print(count)
        if count > 100000:
            break
        count += 1
        TOKEN += jieba_cut(i)
    print(TOKEN[:10])
    words_count = Counter(TOKEN)

    print(words_count.most_common(100))
    print('TOKEN--的长度为:', len(TOKEN))
    print(prob_1('的'))

    TOKEN = [str(t) for t in TOKEN]
    TOKEN_2_GRAM = [''.join(TOKEN[i:i + 2]) for i in range(len(TOKEN[:-2]))]
    print('TOKEN_2_GRAM--的长度为:', len(TOKEN_2_GRAM))
    words_count_2 = Counter(TOKEN_2_GRAM)
    print(prob_2('吴京', '傻'))
    print('句子的概率')
    print('句子：小明去背景的概率为', prob_sentence('小明去背景'))
    print('句子：我们去上班的概率为', prob_sentence('中国人爱看电影'))
