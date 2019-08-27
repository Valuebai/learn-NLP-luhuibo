#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：LuckyHuibo
@Date   ：2019/8/23 15:33
@Desc   ：
=================================================='''
import os
from pyltp import Segmentor
from pyltp import SentenceSplitter
from pyltp import Postagger
from pyltp import NamedEntityRecognizer
from pyltp import Parser
from pyltp import SementicRoleLabeller
import jieba
from collections import defaultdict

# 初始化路径
# ltp模型目录的路径
LTP_DATA_DIR = './data/ltp_data_v3.4.0'

# 分词模型路径，模型名称为`cws.model`
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
# 词性标注模型路径，模型名称为`pos.model`
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
# 命名实体识别模型路径，模型名称为`pos.model`
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')
# 依存句法分析模型路径，模型名称为`parser.model`
par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')
# 语义角色标注模型目录路径，模型目录为`srl`
srl_model_path = os.path.join(LTP_DATA_DIR, 'pisrl.model')


# 加载测试用文本
def load_text(text_path):
    """
    :param text_path: 文本路径
    :return: 文本内容，type=string
    """
    sentence = ''
    with open(text_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if line.strip():
                sentence += line.strip()
    return sentence


# 分句
def pyltp_sentence_splitter(sentences):
    """
    :param sentences: 文本内容，type=string
    :return: 句子列表

    \xa0 是不间断空白符 &nbsp;
    我们通常所用的空格是 \x20 ，是在标准ASCII可见字符 0x20~0x7e 范围内。
    而 \xa0 属于 latin1 （ISO/IEC_8859-1）中的扩展字符集字符，代表空白符nbsp(non-breaking space)。
    latin1 字符集向下兼容 ASCII （ 0x20~0x7e ）。通常我们见到的字符多数是 latin1 的，比如在 MySQL 数据库中。

    去除\xa0    >>>     str.replace(u'\xa0', u' ')

    \u3000 是全角的空白符
    根据Unicode编码标准及其基本多语言面的定义， \u3000 属于CJK字符的CJK标点符号区块内，是空白字符之一。
    它的名字是 Ideographic Space ，有人译作表意字空格、象形字空格等。顾名思义，就是全角的 CJK 空格。
    它跟 nbsp 不一样，是可以被换行间断的。常用于制造缩进， wiki 还说用于抬头，但没见过。

    去除\u3000    >>>     str.replace(u'\u3000',u' ')

    去除空格和\xa0、\u3000    >>>     title.strip().replace(u'\u3000', u' ').replace(u'\xa0', u' ')
    """
    sents_list = []
    sents = SentenceSplitter.split(sentences)

    for sent in sents:
        if sent.strip():
            clean_sent = sent.strip().replace(u'\u3000', u' ').replace(u'\xa0', u' ')
            sents_list.append(clean_sent)
    return sents_list


# 分词1: ltp

def pyltp_cut_words(sentence):
    """
    :param sentence: 文本内容，type=string
    :return: 单词列表
    """
    segmentor = Segmentor()  # 初始化实例
    segmentor.load(cws_model_path)  # 加载模型
    words = segmentor.segment(sentence)  # 分词
    # print ('\t'.join(words))
    segmentor.release()  # 释放模型
    return list(words)


# 分词2: jieba
def jieba_cut_words(sentence):
    words = jieba.cut(sentence)
    return list(words)


# 词性标注
def pyltp_postagger(words):
    postagger = Postagger()  # 初始化实例
    postagger.load(pos_model_path)  # 加载模型

    result = postagger.postag(words)  # 词性标注

    # print ('\t'.join(postags))
    postagger.release()  # 释放模型
    return list(result)


# 命名实体识别
def pyltp_ner(words, postags):
    recognizer = NamedEntityRecognizer()  # 初始化实例
    recognizer.load(ner_model_path)  # 加载模型

    result = recognizer.recognize(words, postags)  # 命名实体识别

    # print ('\t'.join(netags))
    recognizer.release()  # 释放模型
    return list(result)


# 依存句法分析
def pyltp_parsing(words, postags):
    parser = Parser()  # 初始化实例
    parser.load(par_model_path)  # 加载模型
    arcs = parser.parse(words, postags)  # 句法分析

    # print ("\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs)) # 提取依存父节点id
    # rely_id = [arc.head for arc in arcs]  # 提取依存父节点id
    # relation = [arc.relation for arc in arcs]  # 提取依存关系
    parser.release()  # 释放模型
    # return arcs
    return [(arc.head, arc.relation) for arc in arcs]


# 语义角色标注
def pyltp_role_parsing(words, postags, arcs):
    labeller = SementicRoleLabeller()  # 初始化实例
    labeller.load(srl_model_path)  # 加载模型
    roles = labeller.label(words, postags, arcs)  # 语义角色标注
    # for role in roles:
    #    print (words[role.index], "".join(
    #        ["%s:(%d,%d)" % (arg.name, arg.range.start, arg.range.end) for arg in role.arguments]))
    labeller.release()  # 释放模型
    return roles


# 加载"说"的相关词
def load_saying_words(saying_words_path):
    saying_list = []
    with open(saying_words_path, 'r') as f:
        for line in f.readlines():
            if line.strip():
                saying_list.append(line.strip())
    # print(saying_list)
    # print(len(saying_list))
    return saying_list


# 匹配句子中的"说"
def match_saying_words(words, saying_list, postags):
    match = []
    for index, word in enumerate(words):
        # 根据词性标注筛选出动词"说"
        if word in saying_list and postags[index] == 'v':
            match.append((word, index))
    return match


# 提取全文的人名和组织名
def get_total_names(sents_list):
    # 初始化人名字典
    total_names = defaultdict(list)

    for sentence in sents_list:
        # 分词
        words = pyltp_cut_words(sentence)
        # 词性标注
        postags = pyltp_postagger(words)
        # NER
        netags = pyltp_ner(words, postags)

        for start_index, ner in enumerate(netags):
            # 单个词构成的人名
            if ner == 'S-Nh':
                name = words[start_index]
                name_seg = [name]
                total_names[name] = name_seg

            # 多个词构成的人名
            if ner == 'B-Nh':
                end_netags = netags[start_index:]
                end_index = start_index + end_netags.index('E-Nh')
                name = ''.join(words[start_index: end_index + 1])
                name_seg = words[start_index: end_index + 1]
                total_names[name] = name_seg

            # 单个词构成的组织名
            if ner == 'S-Ni':
                name = words[start_index]
                name_seg = [name]
                total_names[name] = name_seg

            # 多个词构成的组织名
            if ner == 'B-Ni':
                end_netags = netags[start_index:]
                end_index = start_index + end_netags.index('E-Ni')
                name = ''.join(words[start_index: end_index + 1])
                name_seg = words[start_index: end_index + 1]
                total_names[name] = name_seg

            # 单个词构成的地名
            # if ner == 'S-Ns':
            #     name = words[start_index]
            #     name_seg = [name]
            #     total_names[name] = name_seg
            #
            # 多个词构成的地名
            # if ner == 'B-Ns':
            #     end_netags = netags[start_index:]
            #     end_index = start_index + end_netags.index('E-Ns')
            #     name = ''.join(words[start_index: end_index + 1])
            #     name_seg = words[start_index: end_index + 1]
            #     total_names[name] = name_seg

    # 返回人名字典，key=人名string，value=人名片段list
    return total_names


# # 获取单句的人名和组织名
# def get_names_with_position(words, netags):
#     names_with_position = []
#
#     for start_index, ner in enumerate(netags):
#         # 单个词构成的人名
#         if ner == 'S-Nh':
#             name = words[start_index]
#             position = (start_index, start_index)
#             names_with_position.append((name, position))
#
#         # 多个词构成的人名
#         if ner == 'B-Nh':
#             end_netags = netags[start_index:]
#             end_index = start_index + end_netags.index('E-Nh')
#             name = ''.join(words[start_index: end_index + 1])
#             position = (start_index, end_index)
#             names_with_position.append((name, position))
#
#         # 单个词构成的组织名
#         if ner == 'S-Ni':
#             name = words[start_index]
#             position = (start_index, start_index)
#             names_with_position.append((name, position))
#
#         # 多个词构成的组织名
#         if ner == 'B-Ni':
#             end_netags = netags[start_index:]
#             end_index = start_index + end_netags.index('E-Ni')
#             name = ''.join(words[start_index: end_index + 1])
#             position = (start_index, end_index)
#             names_with_position.append((name, position))
#
#     return names_with_position


# 优化单句中人名提取函数，可以减少通过NER提取人名失败的情况
# 获取单句的人名和组织名
def get_names_with_position(sentence, words, total_names):
    # 初始化返回列表
    names_with_position = []

    # 遍历全文的人名
    for name, name_seg in total_names.items():
        # 当前句包含人名
        if name in sentence:
            # 单个词构成的人名
            if len(name_seg) == 1:
                # 遍历当前句
                for index, w in enumerate(words):
                    # 提取所有单词构成的人名及其位置
                    if w == name:
                        position = (index, index)
                        names_with_position.append((name, position))

            # 多个词构成的人名
            if len(name_seg) > 1:
                # 遍历当前句
                for start_index, w in enumerate(words):
                    # 提取所有多词构成的人名及其位置
                    if w == name_seg[0]:
                        end_index = start_index + len(name_seg) - 1
                        position = (start_index, end_index)
                        names_with_position.append((name, position))

    # 返回人名和位置结果列表，元素为元组(name, (start_index, end_index))
    return names_with_position


# 获取言论
def get_opinions(words, match_words, names_with_position, arcs):
    # 初始化返回列表
    opinions = []

    # 第一步，根据句子中的人名获取对应言论
    # 遍历句子中的所有人名
    for (name, position) in names_with_position:
        # 获得人名的位置
        start_index = position[0]
        end_index = position[1]

        # 通过人名的 arc.relation = 'SBV' 找言论
        if arcs[end_index][1] == 'SBV':
            # 获取言论初始位置，假设为 SBV 指向的 head 位置
            opinion_index = arcs[end_index][0]

            # 判断言论初始位置是否为标点符号或 arc.relation = 'RAD'
            # 如果是，则其位置往下顺延一位
            if arcs[opinion_index][1] == 'WP' or 'RAD':
                opinion_index += 1

            # 获取完整言论并保存
            opinion = ''.join(words[opinion_index:])
            opinions.append((name, opinion))

            # 获取当前言论的"说"
            # 获取"说"的位置，假设在人名末尾后一位
            saying_index = end_index + 1
            # 获取"说"
            saying = words[saying_index]

            # 将获取的"说"及其位置在"说"的列表中进行匹配
            if (saying, saying_index) in match_words:
                # 将匹配成功的"说"从列表中移除，避免二次匹配
                match_words.remove((saying, saying_index))

    # 第二步，根据句子中的"说"来找对应人名和言论
    # 遍历所有的"说"
    for (saying, position) in match_words:
        # 获取"说"的主体位置
        head_index = arcs[position][0]
        # 获取"说"的 arc.relation
        saying_relation = arcs[position][1]

        # 当"说"的 arc.relation = 'COO' 并且其 arc.head 指向的位置为自身时
        if saying_relation == 'COO' and head_index == position:
            # 获取新的"说"的位置，假设为当前"说"的前一位
            new_saying_index = head_index - 1

            # 获取新"说"的 arc.head 指向位置
            new_head_index = arcs[new_saying_index][0]
            # 获取新"说"的 arc.relation
            new_saying_relation = arcs[new_saying_index][1]

            # 获取言论的初始位置，假设为当前"说"的下一位
            opinion_index = position + 1

            # 判断言论的起始位置是否为标点符号
            # 如果是，则其位置往下顺延一位
            if arcs[opinion_index][1] == 'WP':
                opinion_index += 1

            # 获取完整言论
            opinion = ''.join(words[opinion_index:])

            # 根据 VOB 找到人名
            if new_saying_relation == 'VOB':
                name = words[new_head_index]

                # 保存人名及其言论
                opinions.append((name, opinion))

        # 当"说"的 arc.relation = 'VOB' 时
        if saying_relation == 'VOB':
            # 获取言论的初始位置，假设为"说"的下一位
            opinion_index = position + 1

            # 判断言论的起始位置是否为标点符号
            # 如果是，则其位置往下顺延一位
            if arcs[opinion_index][1] == 'WP':
                opinion_index += 1

            # 获取完整言论
            opinion = ''.join(words[opinion_index:])

            # if arcs[head_index][1] == 'ATT':
            #     head_index = arcs[head_index][0]

            # 判断人名的位置是否为标点符号
            # 如果是，则其位置往下顺延一位
            if arcs[head_index][1] == 'WP':
                head_index += 1

            if arcs[head_index][1] == 'ATT':
                head_index += 1

            # 获取人名
            name = words[head_index]

            # 保存人名及其言论
            opinions.append((name, opinion))

    # 返回言论结果列表，元素为元组(name, opinion)
    return opinions


# 提取单个句子
def extract_single_sentence(saying_list, sentence, total_names):
    words = pyltp_cut_words(sentence)

    # 获取词性标注
    postags = pyltp_postagger(words)

    # 获取ner
    netags = pyltp_ner(words, postags)

    # 获取依存分析
    arcs = pyltp_parsing(words, postags)

    # 获取当前句子中的"说"
    match_words = match_saying_words(words, saying_list, postags)

    if not match_words and "：" not in sentence:
        return False

    # print("匹配到的说", match_words)
    #
    # print("词性标注情况")
    # print([(str(index) + words[index], pos) for index, pos in enumerate(postags)])
    #
    # print("命名实体识别情况")
    # print([(str(index) + words[index], net) for index, net in enumerate(netags)])
    #
    # print("依存句法分析情况")
    # print([(str(index) + words[index], arc) for index, arc in enumerate(arcs)])

    # 获取当前句子中的人名及其位置
    names_with_position = get_names_with_position(sentence, words, total_names)
    # print('人名:', names_with_position)

    res = get_opinions(words, match_words, names_with_position, arcs)

    return res


if __name__ == "__main__":
    pass
