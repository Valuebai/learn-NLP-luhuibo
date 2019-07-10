#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：LuckyHuibo
@Date   ：2019/7/9 23:44
@Desc   ：
=================================================='''
import requests
import re
import pandas as pd
from bs4 import BeautifulSoup

if __name__ == "__main__":
    # 使用requests的注意带点：
    # 报错：直接使用requests.get()报错啦，requests.exceptions.TooManyRedirects: Exceeded 30 redirects.
    #     [错误代码]beijing_metro = requests.get('https://baike.baidu.com/item/%E5%8C%97%E4%BA%AC%E5%9C%B0%E9%93%81/408485')
    # 因为重定向，而导致headers没有维持，后面通过session去get请求
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36'
    headers = {'User-Agent': user_agent}
    sessions = requests.session()
    sessions.headers = headers
    response = sessions.get('https://baike.baidu.com/item/%E5%8C%97%E4%BA%AC%E5%9C%B0%E9%93%81/408485',
                            allow_redirects=False)
    # 解决用requests请求出现乱码
    response.encoding = 'utf-8'
    r = response.text
    print(response)
    # what_we_want = r'<a href="https://(movie\.douban\.com/subject/\d+/\?from=subject-page)" class="" '
    # HTML标记的正则表达式：<(\S*?)[^>]*>.*?|<.*? /> ( 首尾空白字符的正则表达式：^\s*|\s*$或(^\s*)|(\s*$)
    # (可以用来删除行首行尾的空白字符(包括空格、制表符、换页符等等)，非常有用的表达式)

    # 在打印response的html文本后，发现是类似这样的<a href="/item/%E5%8C%97%E4%BA%AC%E5%9C%B0%E9%93%814%E5%8F%B7%E7%BA%BF" target="_blank">
    # 而不是在网页看到的    # <a target="_blank" href = "/item/%E5%8C%97%E4%BA%AC%E5%9C%B0%E9%93%812%E5%8F%B7%E7%BA%BF"> 北京地铁2号线 </a>
    # 故使用beautifulsoup定位到table，缩小范围
    # 最后，用for需要给链接拼接上网址的前缀

    bs = BeautifulSoup(r, "html.parser")
    lines = bs.find_all(target='_blank', name='a', href=re.compile('/item/.{30,100}'),
                        text=re.compile('北京[\d]?\w+[\d]?线'))
    beausoup = BeautifulSoup(str(lines), "html.parser")
    metro_pattern = r'<a href="(/item/.+)" target="_blank">|<a data-lemmaid="\d+" href="(/item/.+)" target="_blank">'
    pattern = re.compile(metro_pattern)
    print('=' * 20)
    beijing = [pattern.findall(str(i)) for i in lines]
    result = set()
    for i in beijing:
        i = ['https://baike.baidu.com' + ''.join(j) for j in i]
        result.add(str(i))
    for i in result:
        print(i)
        # tables = bs.select('table')
    # = list(r'https://baike.baidu.com' + str(beijing_metro[i]))

    # beijing_metro = pattern.findall(str(tables[0]))
    # print(type(beijing_metro[0]))
