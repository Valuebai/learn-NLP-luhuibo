#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：LuckyHuibo
@Date   ：2019/7/31 18:36
@Desc   ：
1.什么是one-hot编码?

独热码，在英文文献中称做 one-hot code, 直观来说就是有多少个状态就有多少比特，而且只有一个比特为1，其他全为0的一种码制

one-hot编码就是保证每个样本中的单个特征只有1位处于状态1，其他的都是0，例如

           1 -> 0001;  2 -> 0010;   3 -> 0100;   4 -> 1000;

2、one-hot在提取文本特征中的应用
one hot在特征提取上属于词袋模型（bag of words）。关于如何使用one-hot抽取文本特征向量我们通过以下例子来说明。假设我们的语料库中有三段话：

我爱中国
爸爸妈妈爱我
爸爸妈妈爱中国

首先对上面语料进行分词，并获取其中的所有的词，然后对每个词进行编号：

              1 我；    2 爱；   3 爸爸；   4 妈妈；   5 中国

然后使用one-hot对每段话提取特征向量：
因此我们得到了最终的特征向量为：
                   我爱中国 -> （ 1，1，0，0，1 ）
                   爸爸妈妈爱我 ->（ 1，1，1，1，0 ）
                   爸爸妈妈爱中国 ->（ 0，1，1，1，1 ）
在实际应用过程中，我们对多篇文本进行分词，并统计词频，生成的词典中词数有几万，十几万，甚至更多，如果都进行one-hot进行编码肯定是行不通的，这时一般会根据词频选取前5K或50K的词进行向量化，摒弃写低频词，提高效率。当然5K或50K对于one-hot编码已经很大了，后面会用word2vec对其进行处理。

=================================================
优缺点分析：

优点：

      一是解决了分类器不好处理离散数据的问题；

     二是在一定程度上也起到了扩充特征的作用（上面样本特征数从3扩展到了9）。

缺点：

      1.  它是一个词袋模型，不考虑词与词之间的顺序，无法保留词序信息；

      2.  它假设词与词相互独立，存在语义鸿沟问题（在大多数情况下，词与词是相互影响的）；

      3.  它得到的特征是离散稀疏的；

      4.  维度灾难：很显然，如果上述例子词典中包含10K个单词，那么每个需要用10000维的向量表示，采用one-hot编码，对角线元素均设为1，其余为0，也就是说除了文本中出现的词语位置不为0，其余9000多的位置均为0，如此高纬度的向量会严重影响计算速度。

3. one-hot编程实现

=================================================='''

# 手动实现one-hot编码
print('=' * 30)
print('手动实现one-hot编码')
import numpy as np

contents = ['我 毕业 于 **大学', '我 就职 于 **公司']
dict = {}
for content in contents:
    for word in content.split():
        if word not in dict:
            dict[word] = len(dict) + 1

results = np.zeros(shape=(len(contents), len(dict) + 1, max(dict.values()) + 1))
# 创建 2 个7*7的矩阵
print('====dict is', dict)
for i, content in enumerate(contents):
    for j, word in list(enumerate(content.split())):
        index = dict.get(word)
        results[i, j, index] = 1
# print(results)
results2 = np.zeros(shape=(len(contents), max(dict.values()) + 1))

for i, content in enumerate(contents):
    for _, word in list(enumerate(content.split())):
        index = dict.get(word)
        results2[i, index] = 1
print(results2)

# Keras中one-hot编码的实现
print('=' * 30)
print('Keras中one-hot编码的实现')
from keras.preprocessing.text import Tokenizer

contents = ['我 毕业 于 **大学', '我 就职 于 **公司']

# 构建单词索引

tokenizer = Tokenizer()
tokenizer.fit_on_texts(contents)

word_index = tokenizer.word_index
print((word_index))
print(len(word_index))

sequences = tokenizer.texts_to_sequences(contents)
print(sequences)

one_hot_result = tokenizer.texts_to_matrix(contents)
print(one_hot_result)

if __name__ == "__main__":
    pass
