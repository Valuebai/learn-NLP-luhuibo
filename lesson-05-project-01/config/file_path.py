# 该文件主要设置file path，统一在这里进行修改
# 使用方法：在其他文件中导入
#
import os

LTP_MODEL_PATH = '../data/ltp_data_v3.4.0'
SYNONYMS_PATH = '../data/synonyms/synonyms.txt'
DEFAULT_STOPWORDS_PATH = '../data/stop_words/stopwords.txt'
WORD2VER_MODEL_PATH = '../data/word2vec100.wv'

path_news_sentences_xut_txt = '../data/news-sentences-xut.txt'
path_news_txt = '../data/news.txt'
path_news_model = '../data/news_model'

# 初始化路径
# ltp模型目录的路径
LTP_DATA_DIR = 'C:\AI-NLP\learn-NLP-luhuibo\lesson-05-project-01\data\ltp_data_v3.4.0'

# 分词模型路径，模型名称为`cws.model`
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
# 词性标注模型路径，模型名称为`pos.model`
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
# 命名实体识别模型路径，模型名称为`pos.model`
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')
# 依存句法分析模型路径，模型名称为`parser.model`
par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')
# 语义角色标注模型目录路径，模型目录为`srl`
srl_model_path = os.path.join(LTP_DATA_DIR, 'pisrl.model')

# said的路径
said_path = 'C:\AI-NLP\learn-NLP-luhuibo\lesson-05-project-01\similar_said'
