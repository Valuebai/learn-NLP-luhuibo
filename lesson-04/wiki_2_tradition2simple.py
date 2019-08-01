#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：LuckyHuibo
@Date   ：2019/8/1 0:59
@Desc   ：
将提取的wiki中文语料txt转化为简体的

不要用opencc处理，麻烦
改用hanziconv，直接pip install hanziconv，再from hanziconv import HanziConv
使用：HanziConv.toSimplified(要转换的str)
=================================================='''

from hanziconv import HanziConv

print('主程序执行开始...')

input_file_name = './data/wiki.cn.txt'
output_file_name = './data/wiki.cn.simple.txt'
input_file = open(input_file_name, 'r', encoding='utf-8')
output_file = open(output_file_name, 'w', encoding='utf-8')

print('开始读入繁体文件...')
lines = input_file.readlines()
print('读入繁体文件结束！')

print('转换程序执行开始...')
count = 1
for line in lines:
    # output_file.write(zhconv.convert(line, 'zh-hans'))
    output_file.write(HanziConv.toSimplified(line))  # 使用hanziconv和zhconv简繁体转换对比
    count += 1
    if count % 10000 == 0:
        print('目前已转换%d条数据' % count)
print('转换程序执行结束！')

print('主程序执行结束！')
