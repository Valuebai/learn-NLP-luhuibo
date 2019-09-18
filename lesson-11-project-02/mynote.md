
## Project-02-文本摘要，项目2的开始
```jupyterpython

## Auto Summarization
#### Main References: 
1. TextRank: https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf
2. SentenceEmbedding: https://openreview.net/pdf?id=SyK00v5xx
3. InformationRetrievalAndTextMining: https://github.com/Artificial-Intelligence-for-NLP-Chinese/References/tree/master/information-retrieve

——from Mr.Gao[https://github.com/Computing-Intelligence/AutoSummarization]
```

- word2vec: FastText
- 句子向量：2019年普林斯顿大学的文章，老师的参考论文
- 文章向量
for sentence in sentences_of_document:
    猪ta = distances(sentence, doc)

<font color=#00ffff size=5>神经网络与词向量框架-Word2Vec的结合（里面有用到神经网络，size=100-300）</font>

> <font color=#00ffff size=5>**知识点回顾**：</font>
>> - CBOW 是输入周围的单词，预测中间的单词——Continue Bag of Words
>> - Skip-Ngram 根据中间的，预测周围的单词
>> - softmax

---

- H~ softmax
  - 有层级的softmax

- Huffman Tree
  - 特点，出现频率越高，就离根节点越近
  - 面试题：
  1. 哈弗曼树要怎么构建的呢？？？
  2. 哈弗曼树在词向量中有什么用？
  3. 怎么解决在反向传播的问题
  
- Negative sampling 负采样
  - PPT的图
  - 不太理解 

早期4个比较有名的词向量方法
- CBOW
- Skip-gram

- FastText
  - （新的词向量方式，类似CBOW,skip-gram）
  - 2017年fb提出，在skip-gram上改进提出
  - 应用：阿里巴巴和蚂蚁金服集团
  - 解决sub-word,out of vocabulary问题，大厂里面经常用到FastText
  
- Glove
  - 产生于斯坦福大学
  - 火过一段时间后，现在没怎么用了

后面seq2seq涉及的3个（学完这门课，需要掌握的7个词向量方法）
- cove
- ELMo
- BERT


## 公式的练习

$x_1$

$$数学公式$$

$x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}$
\\(x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}\\)