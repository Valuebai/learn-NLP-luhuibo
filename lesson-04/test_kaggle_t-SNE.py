#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：LuckyHuibo
@Date   ：2019/7/30 18:03
@Desc   ：test Visualizing Word Vectors with t-SNE
test the code of https://www.kaggle.com/jeffd23/visualizing-word-vectors-with-t-sne

SNE is pretty useful when it comes to visualizing similarity between objects. It works by taking a group of
high-dimensional (100 dimensions via Word2Vec) vocabulary word feature vectors, then compresses them down to
2-dimensional x,y coordinate pairs. The idea is to keep similar words close together on the plane,
while maximizing the distance between dissimilar words.

Steps
1. Clean the data
2. Build a corpus
3. Train a Word2Vec Model
4. Visualize t-SNE representations of the most common words
Credit: Some of the code was inspired by this awesome NLP repo.
=================================================='''


if __name__ == "__main__":
    pass
