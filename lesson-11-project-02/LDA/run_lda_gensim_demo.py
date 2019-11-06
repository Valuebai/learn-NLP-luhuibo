#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：LuckyHuibo
@Date   ：2019/10/21 18:45
@Desc   ：
=================================================='''

if __name__ == "__main__":
    # 准备文档集合
    doc1 = "Sugar is bad to consume. My sister likes to have sugar, but not my father."
    doc2 = "My father spends a lot of time driving my sister around to dance practice."
    doc3 = "Doctors suggest that driving may cause increased stress and blood pressure."
    doc4 = "Sometimes I feel pressure to perform well at school, but my father never seems to drive my sister to do better."
    doc5 = "Health experts say that Sugar is not good for your lifestyle."

    # 整合文档数据
    doc_complete = [doc1, doc2, doc3, doc4, doc5]

    # ---
    # 数据清洗和预处理
    # 数据清洗对于任何文本挖掘任务来说都非常重要，在这个任务中，移除标点符号，停用词和标准化语料库（Lemmatizer，对于英文，将词归元）。
    import nltk

    nltk.download('stopwords')
    # from nltk import stopwords  这个会报错，需要使用from nltk.corpus import stopwords
    from nltk.corpus import stopwords
    from nltk.stem.wordnet import WordNetLemmatizer
    import string

    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()


    def clean(doc):
        stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
        punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
        normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
        return normalized


    doc_clean = [clean(doc).split() for doc in doc_complete]
    print('doc_clean:{}'.format(doc_clean))

    """
    准备 Document - Term 矩阵
    语料是由所有的文档组成的，要运行数学模型，将语料转化为矩阵来表达是比较好的方式。
    LDA 模型在整个 DT 矩阵中寻找重复的词语模式。Python 提供了许多很好的库来进行文本挖掘任务，“genism” 是处理文本数据比较好的库。
    下面的代码掩饰如何转换语料为 Document - Term 矩阵：
    """
    import gensim
    from gensim import corpora

    # 创建语料的词语词典，每个单独的词语都会被赋予一个索引
    dictionary = corpora.Dictionary(doc_clean)
    print('dictionary is :', dictionary)

    # 使用上面的词典，将转换文档列表（语料）变成 DT 矩阵
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    print('doc_term_matrix is :', doc_term_matrix)

    """
    构建 LDA 模型
    创建一个 LDA 对象，使用 DT 矩阵进行训练。
    训练需要上面的一些超参数，gensim 模块允许 LDA 模型从训练语料中进行估计，并且从新的文档中获得对主题分布的推断。
    """
    # 使用 gensim 来创建 LDA 模型对象
    Lda = gensim.models.ldamodel.LdaModel

    # 在 DT 矩阵上运行和训练 LDA 模型
    ldamodel = Lda(doc_term_matrix, num_topics=3, id2word=dictionary, passes=50)
    print('ldamodel is :', ldamodel)
    # 输出结果
    print(ldamodel.print_topics(num_topics=3, num_words=3))
    # 每一行包含了主题词和主题词的权重，Topic 1 可以看作为“不良健康习惯”，Topic 3 可以看作 “家庭”。

    """
    
    五、提高主题模型结果的一些方法
主题模型的结果完全取决于特征在语料库中的表示，但是语料通常表示为比较稀疏的文档矩阵，因此减少矩阵的维度可以提升主题模型的结果。

1. 根据词频调整稀疏矩阵
根据频率来分布词，高频词更可能出现在结果中，低频词实际上是语料库中的弱特征，对于词频进行分析，可以决定什么频率的值应该被视为阈值。



2. 根据词性标注 (Part of Speech Tag) 调整稀疏矩阵
比起频率特征，词性特征更关注于上下文的信息。主题模型尝试去映射相近的词作为主题，但是每个词在上下文上有可能重要性不同，比如说介词 “IN” 包含 “within”，“upon”, “except”，基数词 “CD” 包含：许多(many)，若干（several)，个把(a，few)等等，情态助动词 “MD” 包含 “may”，“must” 等等，这些词可能只是语言的支撑词，对实际意义影响不大，因此可以通过词性来消除这些词的影响。



3. 调整 LDA 的 Batch 大小
为了得到主题中最重要的主题词，语料可以被分为固定大小的 batch，在这些 batch 上运行 LDA 模型会提供不同的结果，但是最佳的主题词会在这些 batch 上有交集。



主题模型用于特征选择
比如说文本分类任务中，LDA 可以用来选择特征，因为训练数据中含有类别信息，可以在不同类别的结果中，删除相同的、比较常见的主题词，为主题类别提供更好的特征。
————————————————
版权声明：本文为CSDN博主「情怀丶」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/selinda001/article/details/80446766
    
    """
