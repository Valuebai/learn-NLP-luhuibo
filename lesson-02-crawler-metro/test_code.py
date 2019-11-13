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
import json

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
    # print(response.text)

    """ 第一次使用，发现有问题，用下面的方案，此次记录
    
    # 在打印response的html文本后，发现是类似这样的<a href="/item/%E5%8C%97%E4%BA%AC%E5%9C%B0%E9%93%814%E5%8F%B7%E7%BA%BF" target="_blank">
    # 而不是在网页看到的    # <a target="_blank" href = "/item/%E5%8C%97%E4%BA%AC%E5%9C%B0%E9%93%812%E5%8F%B7%E7%BA%BF"> 北京地铁2号线 </a>
    # 故使用beautifulsoup定位到table，缩小范围
    # 最后，用for需要给链接拼接上网址的前缀

    soup = BeautifulSoup(response.text, "html.parser")
    lines = soup.find_all(target='_blank', name='a', href=re.compile('/item/.{30,100}'),
                          text=re.compile('北京[\d]?\w+[\d]?线'))
    metro_pattern = r'<a href="(/item/.+)" target="_blank">|<a data-lemmaid="\d+" href="(/item/.+)" target="_blank">'
    pattern = re.compile(metro_pattern)
    beijing = [pattern.findall(str(i)) for i in lines]
    result = set()
    for i in beijing:
        i = ['https://baike.baidu.com' + ''.join(j) for j in i]
        result.add(str(i))
        for i in result:
         print(i)
    """

    # 后面上面的数据有问题&太麻烦了！！，重新整理了下思路
    # 1.用requests获取到html
    # 2.用bs 找到需要获取的表格，缩小范围（要获取的表格是第3个[2]）
    # 3.1结合正则表达式，获取到所有的
    # 3.2 再次利用bs，非常的方便找到表格中的数据
    print("===下面是用select的方法找到html中的表格")
    soup = BeautifulSoup(response.text, "html.parser")
    tables = soup.find_all('table')
    soup2 = BeautifulSoup(str(tables[2]), "html.parser")
    for i in soup2.find_all('a'):
        if i.has_attr("href"):
            print(i.string, i['href'])

    print("===对1号线进行处理")
    response = sessions.get('https://baike.baidu.com/item/%E5%8C%97%E4%BA%AC%E5%9C%B0%E9%93%811%E5%8F%B7%E7%BA%BF',
                            allow_redirects=False)
    response.encoding = 'utf-8'
    soup = BeautifulSoup(response.text, "html.parser")
    tables = soup.find_all('table')

    a = re.compile(r'<td align="center" colspan="1" rowspan="1" valign="middle">([\u4e00-\u9fa5]+)</td>').findall(
        str(tables[2]))
    print(a)

    print("===对4号线进行处理")
    response = sessions.get('https://baike.baidu.com/item/%E5%8C%97%E4%BA%AC%E5%9C%B0%E9%93%814%E5%8F%B7%E7%BA%BF',
                            allow_redirects=False)
    response.encoding = 'utf-8'
    soup = BeautifulSoup(response.text, "html.parser")
    tables = soup.find_all('table')
    print(tables[2])
    a = re.compile(r'<td align="center" colspan="1" rowspan="1" valign="middle">([\u4e00-\u9fa5]+)</td>').findall(
        str(tables[2]))
    print(a)

    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36'
    headers = {'User-Agent': user_agent}
    sessions = requests.session()
    sessions.headers = headers
    response = sessions.get('https://zh.wikipedia.org/wiki/%E5%B9%BF%E5%B7%9E%E5%9C%B0%E9%93%81', verify=False)
    print(response)