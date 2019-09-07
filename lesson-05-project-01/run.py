#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：LuckyHuibo
@Date   ：2019/9/3 23:14
@Desc   ：
=================================================='''
from flask import Flask, render_template, request
from similar_said.speechExtract import del_sentences
import jieba
import re

app = Flask(__name__, template_folder='templates', static_folder='static')


def get_fly_words(fly_str):
    fly_words = []
    for x in re.findall(r'\w+', fly_str):
        fly_words += jieba.lcut(x)
    return {
        'wordList': fly_words,
    }


@app.route('/')
def index():
    fly_str = """
        新闻人物言论自动提取。 
        新闻人物言论即是在报道的新闻中，某个人物、团体或机构在某个时间、某个地点表达某种观点、意见或态度。
        面对互联网信息量的不断扩张，用户迫切地需要自动化的信息获取工具来帮助在海量的信息源中迅速找到和获得真正所需的信息。主要相关方面的研究有自动摘要、关键词提取以及人物言论的自动提取，这些都可以帮助用户快速准确的获取其所需的真正信息，节省用户时间，提高用户体验。其中新闻人物言论自动提取就可以帮助用户在新闻阅读、观点总结中能够发挥较大的辅助作用。
    """
    return render_template('index.html', **get_fly_words(fly_str))


@app.route('/extra', methods=['GET', 'POST'])
def extra():
    news = request.form['news']
    print('news is ', news)
    if not news:
        return '<script>alert("没有输入内容！")</script>'
    parse = del_sentences(news)
    print('parse is', parse[0])
    # infos = parse()
    # if isinstance(infos, list):
    #     infos_type = "list"
    # else:
    #     infos_type = 'str'
    return render_template('extra.html', news=parse[0])


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
