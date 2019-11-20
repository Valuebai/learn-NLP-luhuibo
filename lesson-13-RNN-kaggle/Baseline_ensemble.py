#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：Valuebai
@Date   ：2019/11/20 16:56
@Desc   ：

There are two very different strong baselines currently in the kernels for this competition:

- An LSTM model, which uses a recurrent neural network to model state across each text, with no feature engineering
- An NB-SVM inspired model, which uses a simple linear approach on top of naive bayes features

In theory, an ensemble works best when the individual models are as different as possible.
Therefore, we should see that even a simple average of these two models gets a good result.
Let's try it! First, we'll load the outputs of the models (in the Kaggle Kernels environment
you can add these as input files directly from the UI; otherwise you'll need to download them first).
=================================================='''

import numpy as np, pandas as pd

path_input = './input/'
path_output = './output/'
comp = 'jigsaw-toxic-comment-classification-challenge/'

f_lstm = f'{path_output}baseline2_submission.csv'
f_nbsvm = f'{path_output}baseline3_submission.csv'


p_lstm = pd.read_csv(f_lstm)
p_nbsvm = pd.read_csv(f_nbsvm)

# Now we can take the average of the label columns.
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
p_res = p_lstm.copy()
p_res[label_cols] = (p_nbsvm[label_cols] + p_lstm[label_cols]) / 2

# And finally, create our CSV.

p_res.to_csv(f'{path_output}Baseline_ensemble_submission.csv', index=False)
print(f'saving {path_output}Baseline_ensemble_submission.csv success!')

# As we hoped, when we submit this to Kaggle, we get a great result - much better than the
# individual scores of the kernels we based off. This is currently the best Kaggle kernel submission
# that runs within the kernels sandbox!