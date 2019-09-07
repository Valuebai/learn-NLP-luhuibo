import re
import jieba


def cut(string):
    return ' '.join(jieba.cut(string))


def token(string):
    string = re.findall(
        '[\d|\w|\u3002 |\uff1f |\uff01 |\uff0c |\u3001 |\uff1b |\uff1a |\u201c |\u201d |\u2018 |\u2019 |\uff08 |\uff09 |\u300a |\u300b |\u3008 |\u3009 |\u3010 |\u3011 |\u300e |\u300f |\u300c |\u300d |\ufe43 |\ufe44 |\u3014 |\u3015 |\u2026 |\u2014 |\uff5e |\ufe4f |\uffe5]+',
        string)
    return ' '.join(string)


def deal(string):
    string = token(string)
    return cut(string)


if __name__ == '__main__':
    # string = '大家好， 我在学习NLP'
    string = """
        新华社阿布扎比5月27日电（记者苏小坡）阿联酋阿布扎比王储穆罕默德26日晚在首都阿布扎比会见来访的苏丹过渡军事委员会主席阿卜杜勒·法塔赫·布尔汉时表示，阿联酋将支持苏丹为维护国家安全和稳定所做出的努力，并呼吁各方通过对话实现民族和解。

    据阿联酋通讯社报道，穆罕默德表示，相信苏丹有能力克服目前的困难，实现和平的政治过渡和民族和解。

    布尔汉对阿联酋支持苏丹的立场表示感谢，并特别感谢阿联酋对苏丹提供的财政帮助。

    4月21日，阿联酋和沙特宣布联合向苏丹提供30亿美元实物和现金援助。目前，阿联酋和沙特已提供5亿美元作为苏丹央行的存款。

    4月11日，苏丹国防部长伊本·奥夫宣布推翻巴希尔政权，并成立过渡军事委员会，负责执掌国家事务。4月12日，奥夫宣布辞去其过渡军事委员会主席职务，由布尔汉接任。
        """
    cut_str = deal(string)
    print(type(cut_str))
    print(cut_str)
