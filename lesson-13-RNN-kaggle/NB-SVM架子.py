#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：Valuebai
@Date   ：2019/11/20 18:23
@Desc   ：
=================================================='''
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# NB-SVM架子
class NbSvmClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, dual=False, n_jobs=1):
        self.C = C
        self.dual = dual
        self.n_jobs = n_jobs

    def predict(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict(x.multiply(self._r))

    def predict_proba(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict_proba(x.multiply(self._r))

    def fit(self, x, y):
        # Check that X and y have correct shape
        y = y.values
        x, y = check_X_y(x, y, accept_sparse=True)

        def pr(x, y_i, y):
            p = x[y == y_i].sum(0)
            return (p + 1) / ((y == y_i).sum() + 1)

        self._r = sparse.csr_matrix(np.log(pr(x, 1, y) / pr(x, 0, y)))
        x_nb = x.multiply(self._r)
        self._clf = LogisticRegression(C=self.C, dual=self.dual, n_jobs=self.n_jobs).fit(x_nb, y)
        return self


# ---
path_input = './input/'
path_output = './output/'
comp = 'jigsaw-toxic-comment-classification-challenge/'

train = pd.read_csv(f'{path_input}{comp}train.csv')
test = pd.read_csv(f'{path_input}{comp}test.csv')
subm = pd.read_csv(f'{path_input}{comp}sample_submission.csv')



# We'll create a list of all the labels to predict, and we'll also create a 'none' label so we can see
# how many comments have no labels. We can then summarize the dataset.

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train['none'] = 1 - train[label_cols].max(axis=1)


# There are a few empty comments that we need to get rid of, otherwise sklearn will complain.
COMMENT = 'comment_text'
train[COMMENT].fillna("unknown", inplace=True)
test[COMMENT].fillna("unknown", inplace=True)

'''
---
Building the model
We'll start by creating a bag of words representation, as a term document matrix. 
We'll use ngrams, as suggested in the NBSVM paper.
'''

import re, string

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')


def tokenize(s): return re_tok.sub(r' \1 ', s).split()


'''
It turns out that using TF-IDF gives even better priors than the binarized features used in the paper. 
I don't think this has been mentioned in any paper before, but it improves leaderboard score from 0.59 to 0.55.
'''

n = train.shape[0]
vec = TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenize,
                      min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
                      smooth_idf=1, sublinear_tf=1)
trn_term_doc = vec.fit_transform(train[COMMENT])

x = trn_term_doc
model = NbSvmClassifier(C=4, dual=True, n_jobs=-1).fit(training_features, training_labels)
