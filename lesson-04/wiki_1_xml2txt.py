#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：LuckyHuibo
@Date   ：2019/8/1 0:54
@Desc   ：
=================================================='''

"""

功能：这个代码是将从网络上下载的xml格式的wiki百科训练语料转为txt格式

wiki百科训练语料下载，官网
https://dumps.wikimedia.org/zhwiki/20190720/  （我用的是这个，用UC浏览器下载）
https://dumps.wikimedia.org/zhwiki/  （在这里下载最新的）
    链接：https://pan.baidu.com/s/1eLkybiYOE_aVxsN0pALATg
    密码：hmtn
"""

from gensim.corpora import WikiCorpus

if __name__ == '__main__':

    print('主程序开始...')

    input_file_name = './data/zhwiki-20190720-pages-articles-multistream.xml.bz2'  # 提前建好data文件夹
    output_file_name = './data/wiki.cn.txt'
    print('开始读入wiki数据...')
    input_file = WikiCorpus(input_file_name, lemmatize=False, dictionary={})
    print('wiki数据读入完成！')

    print('处理程序开始...')
    with open(output_file_name, 'w', encoding="utf-8") as f:
        count = 0
        for text in input_file.get_texts():
            f.write(' '.join(text) + '\n')
            count = count + 1
            if count % 10000 == 0:
                print('目前已处理%d条数据' % count)
    print('处理程序结束！')

    print('主程序结束！')
