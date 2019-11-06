#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：LuckyHuibo
@Date   ：2019/10/28 10:31
@Desc   ：
=================================================='''

import jieba
import numpy as np
import collections
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


def split_sentence(text, punctuation_list='!?。！？'):
    """
    将文本段安装标点符号列表里的符号切分成句子，将所有句子保存在列表里。
    """
    sentence_set = []
    inx_position = 0  # 索引标点符号的位置
    char_position = 0  # 移动字符指针位置
    for char in text:
        char_position += 1
        if char in punctuation_list:
            next_char = list(text[inx_position:char_position + 1]).pop()
            if next_char not in punctuation_list:
                sentence_set.append(text[inx_position:char_position])
                inx_position = char_position
    if inx_position < len(text):
        sentence_set.append(text[inx_position:])

    sentence_with_index = {i: sent for i, sent in
                           enumerate(sentence_set)}  # dict(zip(sentence_set, range(len(sentences))))
    return sentence_set, sentence_with_index


def get_tfidf_matrix(sentence_set, stop_word):
    corpus = []
    for sent in sentence_set:
        sent_cut = jieba.cut(sent)
        sent_list = [word for word in sent_cut if word not in stop_word]
        sent_str = ' '.join(sent_list)
        corpus.append(sent_str)

    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    # word=vectorizer.get_feature_names()
    tfidf_matrix = tfidf.toarray()
    return np.array(tfidf_matrix)


def get_sentence_with_words_weight(tfidf_matrix):
    sentence_with_words_weight = {}
    for i in range(len(tfidf_matrix)):
        sentence_with_words_weight[i] = np.sum(tfidf_matrix[i])

    max_weight = max(sentence_with_words_weight.values())  # 归一化
    min_weight = min(sentence_with_words_weight.values())
    for key in sentence_with_words_weight.keys():
        x = sentence_with_words_weight[key]
        sentence_with_words_weight[key] = (x - min_weight) / (max_weight - min_weight)

    return sentence_with_words_weight


def get_sentence_with_position_weight(sentence_set):
    sentence_with_position_weight = {}
    total_sent = len(sentence_set)
    for i in range(total_sent):
        sentence_with_position_weight[i] = (total_sent - i) / total_sent
    return sentence_with_position_weight


def similarity(sent1, sent2):
    """
    计算余弦相似度
    """
    return np.sum(sent1 * sent2) / 1e-6 + (np.sqrt(np.sum(sent1 * sent1)) * \
                                           np.sqrt(np.sum(sent2 * sent2)))


def get_similarity_weight(tfidf_matrix):
    sentence_score = collections.defaultdict(lambda: 0.)
    for i in range(len(tfidf_matrix)):
        score_i = 0.
        for j in range(len(tfidf_matrix)):
            score_i += similarity(tfidf_matrix[i], tfidf_matrix[j])
        sentence_score[i] = score_i

    max_score = max(sentence_score.values())  # 归一化
    min_score = min(sentence_score.values())
    for key in sentence_score.keys():
        x = sentence_score[key]
        sentence_score[key] = (x - min_score) / (max_score - min_score)

    return sentence_score


def ranking_base_on_weigth(sentence_with_words_weight,
                           sentence_with_position_weight,
                           sentence_score, feature_weight=[1, 1, 1]):
    sentence_weight = collections.defaultdict(lambda: 0.)
    for sent in sentence_score.keys():
        sentence_weight[sent] = feature_weight[0] * sentence_with_words_weight[sent] + \
                                feature_weight[1] * sentence_with_position_weight[sent] + \
                                feature_weight[2] * sentence_score[sent]

    sort_sent_weight = sorted(sentence_weight.items(), key=lambda d: d[1], reverse=True)
    return sort_sent_weight


def get_summarization(sentence_with_index, sort_sent_weight, topK_ratio=0.3):
    topK = int(len(sort_sent_weight) * topK_ratio)
    summarization_sent = sorted([sent[0] for sent in sort_sent_weight[:topK]])

    summarization = []
    for i in summarization_sent:
        summarization.append(sentence_with_index[i])

    summary = ''.join(summarization)
    return summary


if __name__ == '__main__':
    # test_text = 'training17.txt'
    # with open(test_text, 'r', encoding='utf-8') as f:
    #     text = f.read()
    text = """
        中新网北京12月1日电(记者 张曦) 30日晚，高圆圆和赵又廷在京举行答谢宴，诸多明星现身捧场，其中包括张杰(微博)、谢娜(微博)夫妇、何炅(微博)、蔡康永(微博)、徐克、张凯丽、黄轩(微博)等。

    30日中午，有媒体曝光高圆圆和赵又廷现身台北桃园机场的照片，照片中两人小动作不断，尽显恩爱。事实上，夫妻俩此行是回女方老家北京举办答谢宴。

    群星捧场 谢娜张杰亮相

    当晚不到7点，两人十指紧扣率先抵达酒店。这间酒店位于北京东三环，里面摆放很多雕塑，文艺气息十足。

    高圆圆身穿粉色外套，看到大批记者在场露出娇羞神色，赵又廷则戴着鸭舌帽，十分淡定，两人快步走进电梯，未接受媒体采访。

    随后，谢娜、何炅也一前一后到场庆贺，并对一对新人表示恭喜。接着蔡康永满脸笑容现身，他直言：“我没有参加台湾婚礼，所以这次觉得蛮开心。”

    曾与赵又廷合作《狄仁杰之神都龙王》的导演徐克则携女助理亮相，面对媒体的长枪短炮，他只大呼“恭喜！恭喜！”

    作为高圆圆的好友，黄轩虽然拍杂志收工较晚，但也赶过来参加答谢宴。问到给新人带什么礼物，他大方拉开外套，展示藏在包里厚厚的红包，并笑言：“封红包吧！”但不愿透露具体数额。

    值得一提的是，当晚10点，张杰压轴抵达酒店，他戴着黑色口罩，透露因刚下飞机所以未和妻子谢娜同行。虽然他没有接受采访，但在进电梯后大方向媒体挥手致意。

    《我们结婚吧》主创捧场

    黄海波(微博)获释仍未出席

    在电视剧《咱们结婚吧》里，饰演高圆圆母亲的张凯丽，当晚身穿黄色大衣出席，但只待了一个小时就匆忙离去。

    同样有份参演该剧，并扮演高圆圆男闺蜜的大左(微信号：dazuozone) 也到场助阵，28日，他已在台湾参加两人的盛大婚礼。大左30日晚接受采访时直言当时场面感人，“每个人都哭得稀里哗啦，晚上是吴宗宪(微博)(微信号：wushowzongxian) 主持，现场欢声笑语，讲了好多不能播的事，新人都非常开心”。

    最令人关注的是在这部剧里和高圆圆出演夫妻的黄海波。巧合的是，他刚好于30日收容教育期满，解除收容教育。

    答谢宴细节

    宾客近百人，获赠礼物

    记者了解到，出席高圆圆、赵又廷答谢宴的宾客近百人，其中不少都是女方的高中同学。

    答谢宴位于酒店地下一层，现场安保森严，大批媒体只好在酒店大堂等待。期间有工作人员上来送上喜糖，代两位新人向媒体问好。

    记者注意到，虽然答谢宴于晚上8点开始，但从9点开始就陆续有宾客离开，每个宾客都手持礼物，有宾客大方展示礼盒，只见礼盒上印有两只正在接吻的烫金兔子，不过工作人员迅速赶来，拒绝宾客继续展示。

        """
    stop_word = []
    with open('stopWordList.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            stop_word.append(line.strip())

    sentence_set, sentence_with_index = split_sentence(text, punctuation_list='!?。！？')
    tfidf_matrix = get_tfidf_matrix(sentence_set, stop_word)
    sentence_with_words_weight = get_sentence_with_words_weight(tfidf_matrix)
    sentence_with_position_weight = get_sentence_with_position_weight(sentence_set)
    sentence_score = get_similarity_weight(tfidf_matrix)
    sort_sent_weight = ranking_base_on_weigth(sentence_with_words_weight,
                                              sentence_with_position_weight,
                                              sentence_score, feature_weight=[1, 1, 1])
    summarization = get_summarization(sentence_with_index, sort_sent_weight, topK_ratio=0.3)
    print('summarization:\n', summarization)
