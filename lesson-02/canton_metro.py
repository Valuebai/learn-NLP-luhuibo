#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：LuckyHuibo
@Date   ：2019/7/12 15:48
@Desc   ：
从wiki百科中爬取广州的地铁线路
=================================================='''
import requests
import re
from bs4 import BeautifulSoup
from collections import defaultdict


def get_each_line_station(url):
    """
    获取每一条地铁线的站点信息
    :param url: https://zh.wikipedia.org/wiki/%E5%B9%BF%E5%B7%9E%E5%9C%B0%E9%93%811%E5%8F%B7%E7%BA%BF （如：1号线）
    :return: list列表存储的所有站点信息
    """
    resp = requests.get(url, timeout=30)
    resp.encoding = 'utf-8'
    s = BeautifulSoup(resp.text, "html.parser")
    tab = s.find_all('table', attrs={"class": "wikitable", "align": "center"})
    station_list = re.findall(r'<th><a href="/wiki/.+" title=".+">(.+)</a>', str(tab))

    return station_list


def get_cantion_metro(lines_dict):
    """
    获取广州的所有地铁信息
    :param lines_dict: 输入的字典形式为:{'1号线': 'https:/..','2号线': 'https:/..'}
    :return: {'1号线': ['广州东站', '体育中心', ...],'2号线':[]...}
    """
    for k_name, v_url in lines_dict.items():
        lines_dict[k_name] = get_each_line_station(lines_dict[k_name])
    return lines_dict


def get_all_stations(lines_dict):
    """
    将一个城市的所有地铁站信息存储到一个List中
    :param lines_dict:
    :return:
    """
    stations = set()
    for k_name, v_value in lines_dict.items():
        stations.update(lines_dict[k_name])
    return stations


def get_longitude_latitude(city_info, station):
    """
    利用高德地图查询对应的地铁站经纬度信息，下面的key需要自己去高德官网申请
    https://lbs.amap.com/api/webservice/guide/api/georegeo
    :param city_info: 具体城市的地铁，如：广州市地铁
    :param station: 具体的地铁站名称，如：珠江新城站
    :return: 经纬度
    """
    addr = city_info + station
    print('*要查找的地点：' + addr)
    parameters = {'address': addr, 'key': '98a3444618af14c0f20c601f5a442000'}
    base = 'https://restapi.amap.com/v3/geocode/geo'
    resp = requests.get(base, parameters, timeout=60)  # 超时设置为60s，翻墙开了全局代理会慢点的
    if resp.status_code == 200:
        answer = resp.json()
        x, y = answer['geocodes'][0]['location'].split(',')
        coor = (float(x), float(y))
        print('=' + station + '的坐标是：', coor)
        return coor


def get_station_location(station_connection):
    """
    获取广州市所有地铁站的经纬度信息
    :param station_connection:list
    :return:字典dict
    """
    station_location = {}
    for station in station_connection:
        md_station = station + "站"
        station_location[station] = get_longitude_latitude(city_info='广州市地铁', station=md_station)
    return station_location


def get_station_connetoins(all_lines_dict):
    """
    获取地铁之间的关联信息
    :param all_lines_dict: 传入所有地铁线路的字典
    :return:
    """
    connections = defaultdict(list)
    for s_key in all_lines_dict.keys():  # generate real station network
        for i in range(len(all_lines_dict[s_key])):
            if i == 0:
                connections[all_lines_dict[s_key][i]].append(all_lines_dict[s_key][i + 1])
            elif i == len(all_lines_dict[s_key]) - 1:
                connections[all_lines_dict[s_key][i]].append(all_lines_dict[s_key][i - 1])
            else:
                connections[all_lines_dict[s_key][i]].append(all_lines_dict[s_key][i - 1])
                connections[all_lines_dict[s_key][i]].append(all_lines_dict[s_key][i + 1])
    print('所有相连接站点信息:', len(connections))
    return connections


def search(start, end, all_connection):
    """
    找到2个地铁站的路线路
    :param start:地铁站,str
    :param end:地铁站,str
    :param all_connection:所有相连接站点信息defaultdict
    :return:列表
    """
    pathes = [[start]]
    passed = [start]

    while pathes:
        path = pathes.pop(0)
        frontier = path[-1]
        nxt = all_connection.get(frontier)
        for station in nxt:
            if station in passed:
                continue
            else:
                new_path = path + [station]
                pathes.append(new_path)
                if station == end: return new_path
                passed.append(station)


def pretty_print(lst):
    print("->".join(lst))


def main():
    # 在国内使用requests请求wiki,google等国外的地址，需要设置代理，将小飞机设置为全局代理
    response = requests.get('https://zh.wikipedia.org/wiki/%E5%B9%BF%E5%B7%9E%E5%9C%B0%E9%93%81')
    response.encoding = 'utf-8'
    # 解析wiki的html，得到beautiful对象
    soup = BeautifulSoup(response.text, "html.parser")
    # 查找bs想要获取的表格，查看html代码，表格的头数据是：
    # <table class="wikitable" align="center" style="width: 100%;">，变成soup的find_all是attrs={}
    tables = soup.find_all('table', attrs={"class": "wikitable", "align": "center", "style": "width: 100%;"})
    # print(tables)   #将数据丢到xx.html文件展示
    pattern = re.compile(r'<a.+?href="(/wiki/.+)" title=".+">(.+线)</a>')
    lines = pattern.findall(str(tables))
    print(type(lines))
    print('获取到广州的地铁路线为：')
    for line in lines:
        print(line)
    # 将获取的线路存到字典中
    canton_lines = {}
    for line in lines:
        canton_lines[line[1]] = 'https://zh.wikipedia.org' + line[0]
    print(canton_lines)
    # 获取所有的线路
    all_lines = get_cantion_metro(canton_lines)
    print('=' * 20)
    print(all_lines)
    # 获取所有的地铁站
    all_stations = get_all_stations(all_lines)
    print('广州市地铁站的个数：', len(all_stations))
    print(all_stations)
    # 获取所有的经纬度
    all_locations = get_station_location(all_stations)
    print(len(all_locations))
    print(all_locations)
    # 找到为什么画出来的图少了一些点，因为使用高德地图API，打印出来的经纬度有一些是重复的，这个后面有时间再处理下
    import matplotlib

    # 指定默认字体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['font.family'] = 'sans-serif'
    # 解决负号'-'显示为方块的问题
    matplotlib.rcParams['axes.unicode_minus'] = False
    import matplotlib.pyplot as plt
    import networkx as nx
    subway_graph = nx.Graph()
    subway_graph.add_nodes_from(list(all_locations.keys()))
    nx.draw(subway_graph, all_locations, with_labels=False, node_size=10)
    # 在pycharm中需要添加下面的代码，才能显示
    plt.show()

    # 获取相连接的地铁信息
    all_connection = get_station_connetoins(all_lines)
    print(all_connection)

    # 打印路径
    pretty_print(search(start='广州东站', end='万胜围', all_connection=all_connection))


if __name__ == "__main__":
    main()
