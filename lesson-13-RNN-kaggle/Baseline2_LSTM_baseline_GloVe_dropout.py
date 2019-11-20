#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：Valuebai
@Date   ：2019/11/20 11:56
@Desc   ：
Improved LSTM baseline¶
This kernel is a somewhat improved version of Keras - Bidirectional LSTM baseline along with some additional documentation of the steps. (NB: this notebook has been re-run on the new test set.)

### 3.Baseline2 - Improved LSTM baseline: GloVe + dropout

- 基于baseline1进行优化改进，训练第二个模型
  - glove词向量
  - dropout
- 我的代码命名为：**Baseline2_LSTM_baseline_GloVe_dropout.py**
- 训练结果：519s 4ms/step - loss: 0.0451 - acc: 0.9830 - val_loss: 0.0465 - val_acc: 0.9829
  - output: baseline2_submission.csv

- 下面的链接是大神公开的
> https://www.kaggle.com/jhoward/improved-lstm-baseline-glove-dropout


- 在kaggle比赛中，常常看到glove.6B的词向量，常见的有50d，100d，200d，300d常用英文单词的词向量
> 可以从https://nlp.stanford.edu/projects/glove/上下载
> 如果下载比较慢，可以从这下载，链接：https://pan.baidu.com/s/1m5zKaJGFwV1VNTsHgGqwRw，提取码：5knd

=================================================='''
import sys, os, re, csv, codecs, numpy as np, pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers

path_input = './input/'
path_output = './output/'
comp = 'jigsaw-toxic-comment-classification-challenge/'
EMBEDDING_FILE = f'{path_input}glove6b50d/glove.6B.50d.txt'
TRAIN_DATA_FILE = f'{path_input}{comp}train.csv'
TEST_DATA_FILE = f'{path_input}{comp}test.csv'

# Set some basic config parameters:
embed_size = 50  # how big is each word vector
max_features = 20000  # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100  # max number of words in a comment to use

# Read in our data and replace missing values:
train = pd.read_csv(TRAIN_DATA_FILE)
test = pd.read_csv(TEST_DATA_FILE)

list_sentences_train = train["comment_text"].fillna("_na_").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("_na_").values

# Standard keras preprocessing, to turn each comment into a list of word indexes of equal length (with truncation or padding as needed).
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)


# Read the glove word vectors (space delimited strings) into a dictionary from word->vector.
def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')


embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE, encoding='utf-8'))

# Use these vectors to create our embedding matrix, with random initialization for words that aren't in GloVe. We'll use the same mean and stdev of embeddings the GloVe has when generating the random init.
all_embs = np.stack(embeddings_index.values())
emb_mean, emb_std = all_embs.mean(), all_embs.std()
emb_mean, emb_std

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

# Simple bidirectional LSTM with two fully connected layers. We add some dropout to the LSTM since even 2 epochs is enough to overfit.

inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
x = GlobalMaxPool1D()(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(6, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Now we're ready to fit out model! Use validation_split when not submitting.
model.fit(X_t, y, batch_size=32, epochs=2, validation_split=0.1)

# And finally, get predictions for the test set and prepare a submission CSV:
y_test = model.predict([X_te], batch_size=1024, verbose=1)
sample_submission = pd.read_csv(f'{path_input}{comp}sample_submission.csv')
sample_submission[list_classes] = y_test
sample_submission.to_csv(f'{path_output}baseline2_submission.csv', index=False)
