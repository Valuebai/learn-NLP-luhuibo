#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：LuckyHuibo
@Date   ：2019/8/5 11:12
@Desc   ：使用中文维基百科语料库训练一个word2vec模型
根据这个教程改编的
https://blog.csdn.net/sinat_29957455/article/details/81432846
1、我已经用WikiExtractor.py提取完数据，并用opencc转为简体字，放在我本地的./data/目录下
2、代码中的文件路径，记得自己修改，文件太大我设置gitignore了
3、stopwords.txt停用词表，我放在同一目录下



中文维基百科下载地址：
https://dumps.wikimedia.org/zhwiki/20190720/  （我用的是这个，用UC浏览器下载）
https://dumps.wikimedia.org/zhwiki/  （在这里下载最新的）

==================================================
'''
# 我碰到的问题：
# 1、用WikiExtractor.py提取后的路径：C:\Users\中文\Desktop\AI-NLP\learn-NLP-luhuibo\lesson-04\zhwiki500\AA\wiki_00
#
# 2、在这个路径执行：
# C:\Users\中文\Desktop\AI-NLP\learn-NLP-luhuibo\lesson-04\zhwiki500\AA>opencc -i wiki_00 -o zh_wiki_00 -c C:\Users\中文\Desktop\AI-NLP\learn-NLP-luhuibo\lesson-04\opencc-1.0.4\share\opencc\t2s.json
# ——报错：t2s.json not found or not accessible.
#
# 在路径执行：C:\Users\壹心理\Desktop\AI-NLP\learn-NLP-luhuibo\lesson-04>opencc -i wiki_00 -o zh_wiki_00 -c opencc-1.0.4\share\opencc\t2s.json
# ——上面这个命令是OK的
#
# # 解决方法（t2s.json not found or not accessible.）：
#
# 下载完成之后，解压到本地即可。解压之后可以将OpenCC下的bin目录添加到系统环境变量中。
# ——不要放到中文路径，直接放到C盘下，C:\opencc-1.0.4\share\opencc\t2s.json
#
# 正常的命令是：opencc -i wiki_00 -o zh_wiki_00 -c C:\opencc-1.0.4\share\opencc\t2s.json


import logging, jieba, os, re


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


def parse_zhwiki(read_file_path, save_file_path):
    '''
    使用正则表达式解析文本
    '''
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
    '''
    将维基百科语料库进行分类
    '''
    zhwiki_path = "./data/"  # 加载zhwiki的路径
    save_path = "./data/"  # 保存zhwiki的路径
    for i in range(3):
        file_path = os.path.join(zhwiki_path, str("zh_wiki_0%s" % str(i)))
        parse_zhwiki(file_path, os.path.join(save_path, "wiki_corpus0%s" % str(i)))


def merge_corpus():
    '''
    合并分词后的文件
    '''
    output = open("./data/wiki_corpus", "w", encoding="utf-8")
    input = "./data/"
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
    generate_corpus()
    merge_corpus()
