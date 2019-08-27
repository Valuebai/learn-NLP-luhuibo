#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：LuckyHuibo
@Date   ：2019/8/22 18:35
@Desc   ：

命名实体结果如下:
ltp命名实体类型为：人名（Nh），地名（NS），机构名（Ni）；
ltp采用BIESO标注体系。B表示实体开始词，I表示实体中间词，E表示实体结束词，S表示单独成实体，O表示不构成实体。

# 欧几里得 - nh - 人名
# 是 - v - 动词
# 西元前 - nt - 时间名词
# 三 - m - 数字
# 世纪 - n - 普通名词
# 的 - u - 助词
# 希腊 - ns - 地理名词
# 数学家- n - 普通名词
# 。 - wp - 标点符号
=================================================='''

from pyltp import SentenceSplitter
from pyltp import Segmentor
from pyltp import NamedEntityRecognizer
from pyltp import Parser
from pyltp import Postagger

import os

if __name__ == "__main__":
    ##分句 - SentenceSplitter
    from pyltp import SentenceSplitter

    sentence = SentenceSplitter.split('我是逗号，我是句号。我是问号？我是感叹号！')
    print('\n'.join(sentence))
    print('-' * 50)

    ## 分词 Segmentor
    LTP_DATA_DIR = './data/ltp_data_v3.4.0'  # ltp模型目录的路径
    cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`

    from pyltp import Segmentor

    segmentor = Segmentor()  # 初始化实例
    segmentor.load(cws_model_path)  # 加载模型
    # pyltp分词支持用户使用自定义词典。分词外部词典本身是一个文本文件
    # segmentor.load_with_lexicon(cws_model_path, '/path/to/your/lexicon') # 加载模型，参数lexicon是自定义词典的文件路径
    words = segmentor.segment('欧几里得是西元前三世纪的希腊数学家。')  # 分词

    print(' '.join(words))
    segmentor.release()  # 释放模型
    print('-' * 50)

    ## 词性标注-Postagger
    pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`

    from pyltp import Postagger

    postagger = Postagger()  # 初始化实例
    postagger.load(pos_model_path)  # 加载模型

    words = ['欧几里得', '是', '西元前', '三', '世纪', '的', '希腊', '数学家', '。']
    postags = postagger.postag(words)  # 词性标注

    print(' '.join(postags))
    postagger.release()  # 释放模型
    print('-' * 50)

    ## 命名实体识别 - NamedEntityRecognizer
    from pyltp import NamedEntityRecognizer

    ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # 命名实体识别模型路径，模型名称为`ner.model`

    recognizer = NamedEntityRecognizer()  # 初始化实例
    recognizer.load(ner_model_path)  # 加载模型

    words = ['欧几里得', '是', '西元前', '三', '世纪', '的', '希腊', '数学家', '。']
    postags = ['nh', 'v', 'nt', 'm', 'n', 'u', 'ns', 'n', 'wp']
    nertags = recognizer.recognize(words, postags)  # 命名实体识别

    print(' '.join(nertags))
    recognizer.release()  # 释放模型
    print('-' * 50)

    ##依存句法分析 - Parser
    par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`

    from pyltp import Parser

    parser = Parser()  # 初始化实例
    parser.load(par_model_path)  # 加载模型

    words = ['欧几里得', '是', '西元前', '三', '世纪', '的', '希腊', '数学家', '。']
    postags = ['nh', 'v', 'nt', 'm', 'n', 'u', 'ns', 'n', 'wp']
    arcs = parser.parse(words, postags)  # 句法分析

    rely_id = [arc.head for arc in arcs]  # 提取依存父节点id
    relation = [arc.relation for arc in arcs]  # 提取依存关系
    heads = ['Root' if id == 0 else words[id - 1] for id in rely_id]  # 匹配依存父节点词语

    for i in range(len(words)):
        print(relation[i] + '(' + words[i] + ', ' + heads[i] + ')')

    parser.release()  # 释放模型
