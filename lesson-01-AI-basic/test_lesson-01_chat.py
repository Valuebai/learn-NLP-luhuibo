# lesson-01作业的第一部分代码
# 这个是我的学习NLP的笔记&作业
# 第一节课的笔记，一开始有邮件中提到的第3题的解法，
# 我好久没有写代码，不能一下子想出来怎么解题，上了第一节课后才知道原来代码能这么写的
# 还有我必须把CS101的课程学下来，我的python编程基础确实不是很强，需要做题！做题！
# 还有上班的时候，可以坐leetcode的题目，上面的题目也是很有帮助的

import random

bad = """
bad = 时间 地点 人物 状态
时间 = 昨天 | 今天
昨天 = 一天前 | 两天前
地点 = 在北京 | 在广州
人物 = 你 | 我
状态 = 很丧 | 不想说话 | 状态不好
"""

good = """
good = 对象 状态 加油
对象 = 你们 | 我们 | 他们 | 你 | 我 | 他
状态 = 要 | 必须
加油 = 元气满满 | 加油鸭 | 努力努力
"""

simple_grammar = """
sentence => noun_phrase verb_phrase
noun_phrase => Article Adj* noun
Adj* => null | Adj Adj*
verb_phrase => verb noun_phrase
Article =>  一个 | 这个
noun =>   女人 |  篮球 | 桌子 | 小猫
verb => 看着   |  坐在 |  听着 | 看见
Adj =>  蓝色的 | 好看的 | 小小的
"""
# 在西部世界里，一个”人类“的语言可以定义为：
human = """
human = 自己 寻找 活动
自己 = 我 | 俺 | 我们 
寻找 = 找找 | 想找点 
活动 = 乐子 | 玩的
"""

# 一个“接待员”的语言可以定义为

host = """
host = 寒暄 报数 询问 业务相关 结尾 
报数 = 我是 数字 号 ,
数字 = 单个数字 | 数字 单个数字 
单个数字 = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 
寒暄 = 称谓 打招呼 | 打招呼
称谓 = 人称 ,
人称 = 先生 | 女士 | 小朋友
打招呼 = 你好 | 您好 
询问 = 请问你要 | 您需要
业务相关 = 玩玩 具体业务
玩玩 = null
具体业务 = 喝酒 | 打牌 | 打猎 | 赌博
结尾 = 吗？
"""


def create_grammar(grammar_str, line_split='=', value_split=('|')):
    """
    主要将输入的文本转为字典中
    :param grammar:
    :return: grammar
    """
    grammar = {}
    # 读取文本的每一行，按照\n来进行切分，如果有空的行，即不进行处理
    for line in grammar_str.split('\n'):
        if not line.strip(): continue
        key, value = line.split(line_split)
        grammar[key.strip()] = [s.split() for s in value.split(value_split)]
    return grammar
    # 唯一的区别就是我用了strip()来清理key-value中value的值
    # 老师用了split()来做清理——清掉空格后存到列表中的
    # 做笔记！做笔记！做笔记！！！
    # strip()会默认移除掉空格' '和换行符的\n


# 这里为什么能用random.choice，因为上面的数据是：
# 'verb_phrase': [['verb', 'noun_phrase']] ——取出来的是完整的['verb', 'noun_phrase']
# 'Article': [['一个'], ['这个']]——取出来的是['一个']或者另一个
# 故这里可以用字典key-vlaue形态来查询，看value=target是否在字典的key=gram_dict中，如果有的话,继续进行递归切分
def generate(gram_dict, target_str):
    # 1.从字典中的key中对应的value随机取出一个值，
    # 2.如果这个值还是字典的key的话，继续上一步的随机取值 如果target不在字典的key中，则返回target
    if target_str not in gram_dict: return target_str
    expand = []
    for t in random.choice(gram_dict[target_str]):
        expand.append(generate(gram_dict, t))
    # expaned = [generate(gram_dict, t) for t in random.choice(gram_dict[target])]
    # 一开始返回expand列表，会得到一个列表[[['这个'], [['蓝色的'], [['蓝色的'], ['null']]], ['女人']], [['坐在'], [['这个'], ['null'], ['女人']]]],我们可以使用''.join将列表中的字符串连接起来
    # ''.join(expand)打印出来的结果中带有null，再处理下
    return ''.join([e for e in expand if e != 'null'])


def generate_n(gram_dict, target_str, n):
    """
    生成多个对话
    :return:
    """
    result_n_list = []
    for i in range(n):
        result_n_list.append(generate(gram_dict, target_str))
    return '\n'.join(result_n_list)


if __name__ == '__main__':
    # 未使用generate_n()函数前的定义
    # for i in range(1, 10):
    #     print('===============第%d次对话' % i)
    #     print(generate(gram=create_grammar(host, line_split='='), target='host'))
    #     print(generate(gram=create_grammar(human, line_split='='), target='human'))
    #     print(generate(gram=create_grammar(simple_grammar), target='sentence'))
    gram_host = create_grammar(grammar_str=host, line_split='=')
    gram_human = create_grammar(grammar_str=human, line_split='=')
    gram_bad = create_grammar(grammar_str=bad)
    gram_good = create_grammar(grammar_str=good)
    print('good and bad的对话:')
    print(generate(gram_bad, 'bad'))
    print(generate(gram_good, 'good'))
    print('bad' + '-' * 20)
    print(generate_n(gram_bad, 'bad', 5))
    print('good' + '-' * 20)
    print(generate_n(gram_good, 'good', 5))
