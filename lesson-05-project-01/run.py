#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：LuckyHuibo
@Date   ：2019/9/3 23:14
@Desc   ：
=================================================='''
from flask import Flask

app = Flask(__name__, template_folder='templates', static_folder='static')


@app.route('/')
def index():
    return "hello world!"


if __name__ == "__main__":
    app.run(host='0.0.0.0')
