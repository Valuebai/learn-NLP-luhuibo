#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：LuckyHuibo
@Date   ：2019/7/30 18:43
@Desc   ：
=================================================='''

import logging, jieba, os, re


def get_stopwords():
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
    # 加载停用词表
    stopword_set = set()
    with open("../stop_words/stopwords.txt", 'r', encoding="utf-8") as stopwords:
        for stopword in stopwords:
            stopword_set.add(stopword.strip("\n"))
    return stopword_set


def parse_zhwiki(read_file_path, save_file_path):
    """
    使用正则表达式解析文本
    :param read_file_path:
    :param save_file_path:
    :return:
    """
    # 过滤掉<doc>
    regex_str = "[^<doc.*>$]|[^</doc>$]"
    file = open(read_file_path, "r", encoding="utf-8")
    # 写文件
    output = open(save_file_path, "w+", encoding="utf-8")
    content_line = file.readline()
    # 获取停用词表
    stopwords = get_stopwords()
    # 定义一个字符串变量，表示一篇文章的分词结果
    article_contents = ""
    while content_line:
        match_obj = re.match(regex_str, content_line)
        content_line = content_line.strip("\n")
        if len(content_line) > 0:
            if match_obj:
                # 使用jieba进行分词
                words = jieba.cut(content_line, cut_all=False)
                for word in words:
                    if word not in stopwords:
                        article_contents += word + " "
            else:
                if len(article_contents) > 0:
                    output.write(article_contents + "\n")
                    article_contents = ""
        content_line = file.readline()
    output.close()


def generate_corpus():
    """
    将维基百科语料库进行分类
    :return:
    """
    zhwiki_path = "D:/dataset/NLP/zhwiki/AA"
    save_path = "D:/dataset/NLP/zhwiki/AA"
    for i in range(3):
        file_path = os.path.join(zhwiki_path, str("zh_wiki_0%s_jt" % str(i)))
        parse_zhwiki(file_path, os.path.join(save_path, "wiki_corpus0%s" % str(i)))


def merge_corpus():
    """
    合并分词后的文件
    :return:
    """
    output = open("D:/dataset/NLP/zhwiki/AA/wiki_corpus", "w", encoding="utf-8")
    input = "D:/dataset/NLP/zhwiki/AA"
    for i in range(3):
        file_path = os.path.join(input, str("wiki_corpus0%s" % str(i)))
        file = open(file_path, "r", encoding="utf-8")
        line = file.readline()
        while line:
            output.writelines(line)
            line = file.readline()
        file.close()
    output.close()


if __name__ == "__main__":
    pass
