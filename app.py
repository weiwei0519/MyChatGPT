# coding=UTF-8
# Web应用的主体，实现页面交互，服务调用和检测
# 

'''
@File: app.py
@Author: Wei Wei
@Time: 2022/7/30 17:13
@Email: weiwei_519@outlook.com
@Software: PyCharm
'''
from functools import wraps
from flask import Flask, render_template, request, jsonify, make_response
from flask_cors import CORS
import time
import threading
import jieba
import json
import os
import openai


# 定义心跳检测函数
def heartbeat():
    print(time.strftime('%Y-%m-%d %H:%M:%S - heartbeat', time.localtime(time.time())))
    timer = threading.Timer(60, heartbeat)
    timer.start()

# 允许跨域
def allow_cross_domain(fun):
    @wraps(fun)
    def wrapper_fun(*args, **kwargs):
        rst = make_response(fun(*args, **kwargs))
        rst.headers['Access-Control-Allow-Origin'] = '*'
        rst.headers['Access-Control-Allow-Methods'] = 'GET,POST'
        allow_headers = "Referer,Accept,Origin,User-Agent"
        rst.headers['Access-Control-Allow-Headers'] = allow_headers
        return rst
    return wrapper_fun


timer = threading.Timer(60, heartbeat)
timer.start()
app = Flask(__name__, static_url_path="/static")
CORS(app, supports_credentials=True)  # 解决flask跨域问题


@app.route('/message', methods=['POST', 'GET'])
@allow_cross_domain
# """定义应答函数，用于获取输入信息并返回相应的答案"""
def reply():
    # 从请求中获取参数信息
    try:
        req_msg = request.form['msg']
    except Exception:
        req_data = request.get_data()
        req_msg = json.loads(req_data)['msg']

    #调用ChatGPT_API
    try:
        res_msg = openai.Completion.create(
            model="text-davinci-003",  # 这里我们使用的是davinci-003的模型，准确度更高。
            prompt=req_msg,
            temperature=1,
            max_tokens=2000,  # 这里限制的是回答的长度，你可以限制字数，如:写一个300字作文等。
            frequency_penalty=0,
            presence_penalty=0
        )
        print(start_sequence, res_msg["choices"][0]["text"].strip())
    except Exception as exc:  # 捕获异常后打印出来
        print(exc)

    if res_msg == ' ':
        res_msg = '欢迎使用ChatGPT智能问答，请输入您的问题'
    # jsonify:是用于处理序列化json数据的函数，就是将数据组装成json格式返回
    # http://flask.pocoo.org/docs/0.12/api/#module-flask.json
    return jsonify({'text': res_msg["choices"][0]["text"].strip()})


@app.route("/")
def index():
    return render_template("index.html")


# 启动APP
if __name__ == '__main__':
    print("欢迎使用ChatGPT智能问答，请在Q:后面输入你的问题，输入quit退出！")
    openai.api_key = "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"
    start_sequence = "\nA:"
    restart_sequence = "\nQ: "
    app.run(host='0.0.0.0', port=8808)
