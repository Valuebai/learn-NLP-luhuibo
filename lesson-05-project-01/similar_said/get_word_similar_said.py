#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：LuckyHuibo
@Date   ：2019/8/28 13:15
@Desc   ：
=================================================='''
from gensim.models import Word2Vec
from collections import defaultdict
import os

def get_related_words(initial_words, model):
    """
    @initial_words
    @model
    """

    unseen = initial_words

    seen = defaultdict(int)

    max_size = 500

    while unseen and len(seen) < max_size:
        if len(seen) % 50 == 0:
            print('seen length : {}'.format(len(seen)))
        node = unseen.pop(0)

        new_expanding = [w for w, s in model.most_similar(node, topn=20)]

        unseen += new_expanding

        seen[node] += 1
    return seen


def get_words_said(model_path):
    model = Word2Vec.load(model_path)
    related_words = get_related_words(['说', '表示', '认为'], model)
    related_words = sorted(related_words.items(), key=lambda x: x[1], reverse=True)
    said = [i[0] for i in related_words if i[1] >= 1]
    return said


def save_said(path):
    said = get_words_said(path)
    string = '|'.join(said)
    try:
        with open("similar_said.txt", 'w') as f:
            f.write(string)
        return True
    except:
        return False


def load_said(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            string = f.readlines()
            string = string[0].split('|')
            return string


if __name__ == '__main__':
    path = "../data/news_model"
    result = save_said(path)
    if result:
        string = load_said("../data/similar_said.txt")
        print(string)