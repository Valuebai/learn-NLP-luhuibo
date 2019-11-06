

上节课剩下的seq2seq部分



seq2seq -> Attention -> Transformer(self.attention) -> BERT （Bidirectional Encoder Representations from Transformers）

BERT基本成为业内作词向量的标准

怎么用呢？
1. Github ，看PPT中的url（模型已经训练好了）
https://github.com/hanxiao/bert-as-service
2. Fine Tuning （已经训练好或者训练一半的时候，需要你再进行模型的调整）
https://www.kaggle.com/taindow/bert-a-fine-tuning-example

知道原理之后知道模型bu work怎么去改他，还能自己去造一个新的模型，写东西要写出来一样才行的



对话机器人

Outline
1. 概览
- 任务型
- 闲聊
- 问答（IR-Based System 和 KG-Based System）

人为地进行分类，实际上是你中有我，我中有你的态势

20:25卡住了2分钟

闲聊型

重点在问答型上，回顾TF-IDF，基于这个 -> 布尔搜索

为什么本机的搜索很慢，而百度，谷歌搜索很快的呢？

原子操作& |  计算机出厂时就设置好的东西

【想法】run.py加载的路径是最顶层，其他的路径都设置为跟run.py同层级的，APP/AA/BB

代码：

np.where(vector_of_) # 知道哪些点不为0

all的意思


几乎所有的对话机器人，核心都是搜索应用的，Boolean Search是最早的，要是20年前值2个亿（在这基础上，加上pagerank就是现代搜索，谷歌之前都都是基于这个的）

set中也是有位运算中的

老师，这个比如在千万级句子中，查一个词会有多久时间
——瓶颈是单位向量长度的存储与运算
把千万级别01的长度用int('01',2)转换为十进制的，然后再&下，再bin(xx)下——老师实际做过的


老师实话：目前深度学习顶多能解决NLP的问题 2/5

- 任务型NLU,NLG
意图:转账
词槽：{
人名：王二麻子
卡名：工资卡
金额：500元
}

国内有几家银行的问答机器人就是老师负责的
5.



