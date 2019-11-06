

## fasttext

因为leeson-11主要记录project-02的相关内容，故把fasttext记录在这里

**fasttext官网的tutorials -Text Classifier和Word representations一定要看下，里面的调参过程很有收获**
- 官网链接：https://fasttext.cc
- 官网的tutorials：https://fasttext.cc/docs/en/supervised-tutorial.html
- github地址：https://github.com/facebookresearch/fastText

- fasttext模型架构和Word2Vec中的CBOW模型很类似。不同之处在于，**fasttext预测标签**，而CBOW模型预测中间词。

> What is fastText?
>> FastText is an open-source, free, lightweight library that allows users to learn text
>> representations and text classifiers. It works on standard, generic hardware. Models
>> can later be reduced in size to even fit on mobile devices.


facebook已经收集了海量语料，训练好了fasttext的词向量模型，目前已经支持了150多种语言。有需要的读者可以直接下载：
https://github.com/facebookresearch/fastText/blob/master/docs/crawl-vectors.md

> fastText原理及实践（达观数据王江），http://www.52nlp.cn/fasttext