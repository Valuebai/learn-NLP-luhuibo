#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：LuckyHuibo
@Date   ：2019/9/26 10:23
@Desc   ：自动提取摘要-通过tfidf方法
=================================================='''
import os
import jieba
import re
from typing import List
import numpy as np
import collections
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

package_path = os.path.dirname(__file__)
default_stop_words_file_path = os.path.join(package_path, 'data/stop_words.txt')

jieba.add_word('十八大')


class SummaryText:
    """
    自动提取摘要-通过tfidf方法
    """

    def __init__(self, stop_words_file_path: str = default_stop_words_file_path):
        with open(stop_words_file_path, 'r', encoding='utf-8') as file:
            self.stop_words = [word.strip('\n') for word in file]

    def summary_text(self, text: str) -> str:
        """
        通过一段文本转换为关键句子列表
            1、将文本转换为句子
            2、计算词tfidf矩阵
            3、计算句子权重、句子位置矩阵、句子相似度权重
            4、获得权重最高的句子
        :param text: 文本，段落
        :return list: 关键句子
        """
        sentences = self._split_sentences(text)
        tfidf_matrix = self._tfidf_matrix(sentences)
        sentences_weight = self._sentences_weight(tfidf_matrix)
        sentences_position_weight = self._sentences_position_weight(sentences)
        sentences_scores = self._sentences_similarity_weight(tfidf_matrix)
        print(len(sentences_weight), len(sentences_position_weight), len(sentences_scores))
        sentences_rank = self._sentences_rank(sentences_weight, sentences_position_weight, sentences_scores)
        print(sentences_rank)
        sentences_summarization = self._summarization(sentences, sentences_rank)
        return '\n'.join(sentences_summarization)

    def _split_sentences(self, text: str) -> List[str]:
        """
        将一段文本差分成单个句子，保存至列表中。
        根据标点（。？！）作为分割依据
        :param text: 文本
        :return:
        """
        text = re.sub(r'[。|？|！]', '&&', text)
        return [sentence.strip('\n').strip() for sentence in text.split("&&")]

    def _tfidf_matrix(self, sentences: List[str]) -> List:
        """
        构建tfidf矩阵
        :param sentences:
        :return:
        """
        corpus = []
        for sentence in sentences:
            corpus.append(' '.join([word for word in jieba.cut(sentence) if
                                    word not in self.stop_words and len(word) > 1 and word != '\t']))
        vectorizer = CountVectorizer()
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
        return tfidf.toarray()

    def _sentences_weight(self, tfidf_matrix):
        sentences_weight = {index: np.sum(weight) for index, weight in enumerate(tfidf_matrix)}
        max_weight = max(sentences_weight.values())
        min_weight = min(sentences_weight.values())
        interval = max_weight - min_weight
        sentences_weight = {key: (value - interval) / interval for key, value in sentences_weight.items()}
        return sentences_weight

    def _sentences_position_weight(self, sentences):
        """
        计算句子位置权重
        :param sentences:
        :return:
        """
        total_sentences = len(sentences)
        sentences_position_weight = {index: (total_sentences - index) / total_sentences for index in
                                     range(total_sentences)}
        return sentences_position_weight

    def _sentences_similarity_weight(self, tfidf_matrix):
        """
        计算句子相似矩阵
        :param tfidf_matrix: 词tfidf矩阵
        :return:
        """
        senctences_scores = collections.defaultdict(lambda: 0.)
        for index, s1 in enumerate(tfidf_matrix):
            score = np.sum([self._similarity(s1, s2) for s2 in tfidf_matrix])
            senctences_scores[index] = score

        max_score = max(senctences_scores.values())
        min_score = min(senctences_scores.values())
        interval = max_score - min_score
        senctences_scores = {key: (value - min_score) / interval for key, value in senctences_scores.items()}
        return senctences_scores

    def _similarity(self, s1, s2):
        """
        计算句子相似度，余弦相似定律
        :param s1:
        :param s2:
        :return:
        """
        return np.sum(s1 * s2) / (1e-6 + np.sqrt(np.sum(s1 * s1)) * np.sqrt(np.sum(s2 * s2)))

    def _sentences_rank(self, sentences_weight, sentences_position_weight, sentences_scores, feature_weight=[1, 1, 1]):
        """
        计算句子排名
        :param sentences_weight:
        :param sentences_position_weight:
        :param sentences_scores:
        :param feature_weight:
        :return:
        """
        weight = {index: (
                feature_weight[0] * weight + feature_weight[1] * sentences_position_weight[index] + feature_weight[
            2] * sentences_scores[index]) for index, weight in sentences_weight.items()}
        return sorted(weight.items(), key=lambda item: item[1], reverse=True)

    def _summarization(self, sentences, sentences_rank, topN_ratio=0.3):
        """
        生成摘要
        :param sentences:
        :param sentences_rank:
        :param topN_ratio:
        :return:
        """
        topN = int(len(sentences_rank) * topN_ratio)
        sentences_index = sorted([index for index, _ in sentences_rank[:topN]])
        return [sentences[index] for index in sentences_index]


if __name__ == '__main__':
    text = """
    十八大以来的五年，是党和国家发展进程中极不平凡的五年。面对世界经济复苏乏力、局部冲突和动荡频发、全球性问题加剧的外部环境，面对我国经济发展进入新常态等一系列深刻变化，我们坚持稳中求进工作总基调，迎难而上，开拓进取，取得了改革开放和社会主义现代化建设的历史性成就。
    为贯彻十八大精神，党中央召开七次全会，分别就政府机构改革和职能转变、全面深化改革、全面推进依法治国、制定“十三五”规划、全面从严治党等重大问题作出决定和部署。五年来，我们统筹推进“五位一体”总体布局、协调推进“四个全面”战略布局，“十二五”规划胜利完成，“十三五”规划顺利实施，党和国家事业全面开创新局面。
    经济建设取得重大成就。坚定不移贯彻新发展理念，坚决端正发展观念、转变发展方式，发展质量和效益不断提升。经济保持中高速增长，在世界主要国家中名列前茅，国内生产总值从五十四万亿元增长到八十万亿元，稳居世界第二，对世界经济增长贡献率超过百分之三十。供给侧结构性改革深入推进，经济结构不断优化，数字经济等新兴产业蓬勃发展，高铁、公路、桥梁、港口、机场等基础设施建设快速推进。农业现代化稳步推进，粮食生产能力达到一万二千亿斤。城镇化率年均提高一点二个百分点，八千多万农业转移人口成为城镇居民。区域发展协调性增强，“一带一路”建设、京津冀协同发展、长江经济带发展成效显著。创新驱动发展战略大力实施，创新型国家建设成果丰硕，天宫、蛟龙、天眼、悟空、墨子、大飞机等重大科技成果相继问世。南海岛礁建设积极推进。开放型经济新体制逐步健全，对外贸易、对外投资、外汇储备稳居世界前列。
    全面深化改革取得重大突破。蹄疾步稳推进全面深化改革，坚决破除各方面体制机制弊端。改革全面发力、多点突破、纵深推进，着力增强改革系统性、整体性、协同性，压茬拓展改革广度和深度，推出一千五百多项改革举措，重要领域和关键环节改革取得突破性进展，主要领域改革主体框架基本确立。中国特色社会主义制度更加完善，国家治理体系和治理能力现代化水平明显提高，全社会发展活力和创新活力明显增强。
    """

    summary_text = SummaryText()
    senctences = summary_text.summary_text(text)
    print(senctences)
