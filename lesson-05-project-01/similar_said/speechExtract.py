#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
from similar_said.urils import deal
from similar_said.get_word_similar_said import load_said
from config.file_path import LTP_DATA_DIR, said_path


# 分句
def split_sentences(string):
    from pyltp import SentenceSplitter
    sents = SentenceSplitter.split(string)
    sentences = [s for s in sents if len(s) != 0]
    return sentences


# 分词
def split_words(sentences):
    sents = [deal(s) for s in sentences]
    # print(sents)
    return sents


# 词性分析
def get_word_pos(ltp_model_path, sents):
    model_path = ltp_model_path
    pos_model_path = os.path.join(model_path, 'pos.model')
    # print('pos_model_path is ', pos_model_path)
    from pyltp import Postagger
    postagger = Postagger()
    postagger.load(pos_model_path)
    postags = [postagger.postag(words.split()) for words in sents]
    postags = [list(w) for w in postags]
    postagger.release()
    return postags


# 依存句法分析
def dependency_parsing(ltp_model_path, sents, postags, said):
    LTP_DATA_DIR = ltp_model_path  # ltp模型目录的路径
    par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`
    ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # 依存句法分析模型路径，模型名称为`ner.model`

    from pyltp import Parser, NamedEntityRecognizer
    recognizer = NamedEntityRecognizer()  # 初始化实例
    recognizer.load(ner_model_path)  # 加载模型

    parser = Parser()  # 初始化实例
    parser.load(par_model_path)  # 加载模型

    contents = []
    for index in range(len(sents)):
        wo = sents[index].split()
        po = postags[index]

        netags = recognizer.recognize(wo, po)  # 命名实体识别
        netags = list(netags)
        # print(netags)
        if ('S-Nh' not in netags) and ('S-Ni' not in netags) and (
                'S-Ns' not in netags):  # 人名、机构名、地名  当人名、机构名、地名在该句中则进行依存句法分析
            continue

        arcs = parser.parse(wo, po)

        arcs = [(arc.head, arc.relation) for arc in arcs]
        # print(arcs)  #[(2, 'SBV'), (0, 'HED'), (5, 'SBV'), (5, 'ADV'), (2, 'VOB')]
        arcs = [(i, arc) for i, arc in enumerate(arcs) if arc[1] == 'SBV']  # SBV 主谓关系 找出主谓关系的句子
        # print(arcs)  #[(0, (2, 'SBV')), (2, (5, 'SBV'))]
        for arc in arcs:
            verb = arc[1][0]  # 2  5
            subject = arc[0]  # 0  1
            if wo[verb - 1] not in said:  # 如果wo[verb - 1]这个所对应的词语  在已建词表said中，则打印出来
                continue
            # print(wo[subject],wo[verb - 1],''.join(wo[verb:]))
            contents.append((wo[subject], wo[verb - 1], ''.join(wo[verb:])))  # 依次为人物、"说"的近义词、文本
    # parser.release()  # 释放模型
    # recognizer.release()  # 释放模型
    # print(contents)
    return contents


def del_sentences(string):
    path = os.path.join(said_path, "similar_said.txt")
    said = load_said(path)

    ltp_model_path = LTP_DATA_DIR
    sentences = split_sentences(string)
    sents = split_words(sentences)

    postags = get_word_pos(ltp_model_path, sents)

    contents = dependency_parsing(ltp_model_path, sents, postags, said)
    contents_dict = []
    for ones in enumerate(contents):
        result = {'name': ones[1][0], 'trigger': ones[1][1], 'content': ones[1][2]}
        contents_dict.append(result)
    return contents_dict


if __name__ == '__main__':
    string = """
    新华社阿布扎比5月27日电（记者苏小坡）阿联酋阿布扎比王储穆罕默德26日晚在首都阿布扎比会见来访的苏丹过渡军事委员会主席阿卜杜勒·法塔赫·布尔汉时表示，阿联酋将支持苏丹为维护国家安全和稳定所做出的努力，并呼吁各方通过对话实现民族和解。

据阿联酋通讯社报道，穆罕默德表示，相信苏丹有能力克服目前的困难，实现和平的政治过渡和民族和解。

布尔汉对阿联酋支持苏丹的立场表示感谢，并特别感谢阿联酋对苏丹提供的财政帮助。

4月21日，阿联酋和沙特宣布联合向苏丹提供30亿美元实物和现金援助。目前，阿联酋和沙特已提供5亿美元作为苏丹央行的存款。

4月11日，苏丹国防部长伊本·奥夫宣布推翻巴希尔政权，并成立过渡军事委员会，负责执掌国家事务。4月12日，奥夫宣布辞去其过渡军事委员会主席职务，由布尔汉接任。
    """
    string1 = """
    台湾工业总会是岛内最具影响力的工商团体之一，2008年以来，该团体连续12年发表对台当局政策的建言白皮书，集中反映岛内产业界的呼声。

    台湾工业总会指出，2015年的白皮书就特别提到台湾面临“五缺”（缺水、缺电、缺工、缺地、缺人才）困境，使台湾整体投资环境走向崩坏。然而四年过去，“五缺”未见改善，反而劳动法规日益僵化、两岸关系陷入紧张、对外关系更加孤立。该团体质疑，台当局面对每年的建言，“到底听进去多少，又真正改善了几多”？

    围绕当局两岸政策，工总认为，由数据来看，当前大陆不仅是台湾第一大出口市场，亦是第一大进口来源及首位对外投资地，建议台湾当局摒弃两岸对抗思维，在“求同存异”的现实基础上，以“合作”取代“对立”，为台湾多数民众谋福创利。

    工总现任理事长、同时也是台塑企业总裁的王文渊指出，过去几年，两岸关系紧张，不仅影响岛内观光、零售、饭店业及农渔蔬果产品的出口，也使得岛内外企业对投资台湾却步，2020年新任台湾领导人出炉后，应审慎思考两岸问题以及中国大陆市场。

        """

    string2 = """中评社高雄8月6日电（记者　高易伸）中国国民党2020参选人、高雄市长韩国瑜6日市政会议结束后被问到台北市长柯文哲炮轰蔡英文身边的人都贪污一事表示，“首都”市长指控“总统”旁边每个人都贪污，这是何等严重？是多么可怕的事情？蔡虽反驳说“抹黑别人不会让自己更乾净”，但蔡英文要仔细去思考，为什么柯会讲这么严厉的话？ 
    韩国瑜说，前阵子他讲话比较宽厚，说蔡英文放了一大堆“虎豹狮象”抢公家及公营事业，今天柯的指责是赤裸裸的，将来会登录历史记载。一个“总统”不会用人，用的人全是贪官污吏是多么可怕？这对台湾民主政治、清廉政治带来多大伤害？蔡要仔细检讨，如果用的官员都吃台湾人的肉、喝台湾人的血，嘴巴讲要民主改革，对台湾带来的伤害是多么巨大。 
    韩国瑜呼吁，蔡英文真的要静下心来好好检讨，用人是不是出了问题。这些人继续拱妳选“总统”，
    是否想要继续吃香喝辣、贪污舞弊？蔡要严肃思考，妳治理国家的能力是不是出了问题？当台北市长做出这么严重的指控这不是开玩笑的。全国民众特别是长期支持民进党的好朋友，你们一定要清醒过来，台湾民主政治到底要走向清廉，还是继续污秽，这个大家要做出思考。柯Ｐ提出这样的指控，台湾民众不能掉以轻心，认为是政党恶斗。 
    柯文哲昨日抨击蔡英文身边用人，引发论战，但也引起蔡多位幕僚
    不满，要求柯道歉。韩国瑜今跟进柯的发言。针对柯今日创立台湾民众党，上午举行创党大会，韩国瑜也表示，柯成立台民党为台湾民主政治增加新动能、新元素，若能把理念理想阐述得更清楚，且剑及履及去实现，对台湾民主发展一定是好事。 
    韩国瑜希望更多优质有理念的人投入台湾民主政治，事实上三年多来，有心人认真一点看，台湾民主政治真的倒退得一塌糊涂。蔡英文执政3年台湾民主政治倒退30年。可见太多不公不义的事情，柯Ｐ市长愿意投入创立新政党，祝福他，也希望柯能让台湾民主政治走向更好，更真更善更美的境界。 """
    string3 = """国台办表示中国必然统一。会尽最大努力争取和平统一，但绝不承诺放弃使用武力。
    昨天想睡觉"""
    print(del_sentences(string3))
