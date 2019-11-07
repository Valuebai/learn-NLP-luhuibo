#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：LuckyHuibo
@Date   ：2019/10/8 11:30
@Desc   ：
=================================================='''

# 利用TextRank，提取文本摘要
import jieba
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer


def cut_sentence(sentence):
    """
    分句
    :param sentence:
    :return:
    """
    # if not isinstance(sentence, unicode):
    # sentence = sentence.decode('utf-8')
    delimiters = frozenset(u'。！？')
    buf = []
    for ch in sentence:
        buf.append(ch)
        if delimiters.__contains__(ch):
            yield ''.join(buf)
            buf = []
    if buf:
        yield ''.join(buf)


def load_stopwords(path='./stop_words.txt'):
    """
    加载停用词
    :param path:
    :return:
    """
    with open(path, encoding="utf-8") as f:
        # stopwords = filter(lambda x: x, list(map(lambda x: x.strip(), f.readlines())))
        stopwords = list(map(lambda x: x.strip(), f.readlines()))
    stopwords.extend([' ', '\t', '\n'])
    return frozenset(stopwords)

    return filter(lambda x: not stopwords.__contains__(x), jieba.cut(sentence))


def get_abstract(content, size=3):
    """
    利用textrank提取摘要
    :param content:
    :param size:
    :return:
    """
    docs = list(cut_sentence(content))
    tfidf_model = TfidfVectorizer(tokenizer=jieba.cut, stop_words=load_stopwords())
    tfidf_matrix = tfidf_model.fit_transform(docs)
    normalized_matrix = TfidfTransformer().fit_transform(tfidf_matrix)
    similarity = nx.from_scipy_sparse_matrix(normalized_matrix * normalized_matrix.T)
    scores = nx.pagerank(similarity)
    tops = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    size = min(size, len(docs))
    indices = list(map(lambda x: x[0], tops))[:size]
    return list(map(lambda idx: docs[idx], indices))


text = u'''
上午，新京报记者从北京市相关部门召开的新闻发布会上获悉，北京市将于4月16日起正式启动北京市首批积分落户申报。对于申报人“在京连续缴纳社会保险7年及以上”的资格条件，如何认定?是否允许存在补缴记录?根据细则规定，“截至积分落户申报工作启动的上一年度12月31日，申请人应在京连续缴纳社会保险满7年(补缴记录累计不超过5个月)，养老、医疗、失业、工伤、生育各项险种的缴费应符合北京市社会保险相关规定。实际缴费记录应在积分落户申报工作启动的上一年度12月31日前形成，且年度积分落户申报阶段开始前缴费状态正常。”
　　北京市人力社保局积分落户服务中心相关负责人介绍，在京连续缴纳社保7年及以上是根据本市社会保险相关规定确定不同险种的缴纳计算起始时间。考虑到申请人工作调动等情况，会存在个别月份的社会保险费补缴，故规定为补缴记录累计不超过5个月。期间断缴未及时补缴的，即使未超过5个月也不能算为连续缴纳。因银行托收或社保经办机构原因造成断缴，如果在年度积分落户申报启动的上一年度12月31日前及时进行了补缴，将不会计入资格条件补缴月数。
'''
# 读取文件
# f=open(r"D:\temp.txt",encoding="utf-8")
# text=f.read()
# f.close()
for i in get_abstract(text, 3):
    template = 'top sen 3 : {}'
    print(template.format(i))
