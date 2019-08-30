from collections import defaultdict
from gensim.models import Word2Vec  # 词向量

WORD2VEC_MODEL = Word2Vec.load("../data/zhwiki_news.word2vec")


def get_similar_words(init_words, model):
    """
    获得相似的单词
    :param init_words: 初始词列表
    :param model:  word2vec模型
    :return:
    """
    unseen = init_words
    seen = defaultdict(int)

    max_nums = 500
    while unseen and len(seen) < max_nums:
        if len(seen) % 50 == 0:
            print(f'seen nums:{len(seen)}')
        word = unseen.pop(0)

        similar_words = [w for w, p in model.most_similar(word, topn=20)]

        unseen += similar_words

        seen[word] += 1  # 权重？？？？
    return seen


# 获得相近词并保存文本
def save_to_file(words):
    related_words = get_similar_words(words, WORD2VEC_MODEL)
    sorted_related_words = sorted(related_words.items(), key=lambda x: x[1], reverse=True)
    with open('../data/words.txt', 'w', encoding='utf-8') as f:
        res = []
        for word, times in sorted_related_words:
            res.append(word)
        f.write(' '.join(res))


if __name__ == "__main__":
    words = ['说', '表示']
    save_to_file(words)
