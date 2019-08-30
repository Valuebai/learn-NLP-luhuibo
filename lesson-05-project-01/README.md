## 项目一：新闻任务言论自动提取

### 语料库
#### 语料库来源：WIKI和提供的数据库news


+ #### 语料库获取
    + ##### Wiki语料库
        ① 使用维基百科下载中文语料库  
        链接：https://dumps.wikimedia.org/zhwiki/20190720/ 

        ② 抽取语料库的内容     
            链接：https://github.com/attardi/wikiextractor  
            方法1: wikiextractor    
            github上下载 `git clone https://github.com/attardi/wikiextractor.git`   
            进入目录，运行 `python WikiExtractor.py -o zhwiki-20190401-pages-articles.xml.bz 文件名`  
            方法2：gensim WikiCorpus   
            安装gensim，调用即可   
        
    + ##### 新闻语料库  
        此项目使用的是阿里云数据库，远程访问即可  
        ```
        数据库地址（Host）  
        用户名（User）  
        用户密码（Password）   
        数据库名（Database）    
        表名  
        ```
        访问工具：pymysql 或者 sqlalchemy  
        代码：./database/get_data.py 数据量大，处理时间有点长
    + ##### 合并两个语料库,进行词向量训练，方便获取与‘说’相近的词
+ #### 数据预处理、Word2Vec词向量训练 
    具体操作，请访问     
    ```
    https://github.com/huangmgithub/NLP_Course/tree/master/Lesson04
    ``` 
    词向量训练完成后，可获得以下文件：
    ```
    wiki.zh.model
    wiki.zh.model.trainables.syn1neg.npy
    wiki.zh.model.wv.vectors.npy
    wiki.zh.vectors
    ```
+ #### 获取与‘说’相近的词 
    工具：搜索树（广度优先） + 动态规划 + NER

+ #### 抽取新闻人物观点
    工具：pyltp(自然语言处理工具) + TF-IDF(文本相似度) 

    pyltp参考文档:  
    `https://pyltp.readthedocs.io/zh_CN/latest/`  
    sklearn参考文档:    
    `https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html`


> ### 服务器环境部署
    To be Continued......
    
    
    
|- data文件不设置了不提交的
|-- ltp_data_v3.4.0 ，哈工大的ltp.model
|-- news.txt，从数据库中提取的语料

将wiki和news_model的数据整合到一起，即分别加载model，训练跟【说】相似的词