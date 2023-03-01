# coding=UTF-8
# 通过ChatGPT调用OpenAI API

'''
@File: ChatGPT_API
@Author: WeiWei
@Time: 2023/2/8
@Email: weiwei_519@outlook.com
@Software: PyCharm
'''

import os
import openai

print("欢迎使用ChatGPT智能问答，请在Q:后面输入你的问题，输入quit退出！")
openai.api_key = "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"
start_sequence = "\nA:"
restart_sequence = "\nQ: "
while True:
    prompt = input(restart_sequence)
    if prompt == 'quit':
        break
    else:
        try:
            response = openai.Completion.create(
                model="text-davinci-003",  # 这里我们使用的是davinci-003的模型，准确度更高。
                prompt=prompt,
                temperature=1,
                max_tokens=2000,  # 这里限制的是回答的长度，你可以限制字数，如:写一个300字作文等。
                frequency_penalty=0,
                presence_penalty=0
            )
            print(start_sequence, response["choices"][0]["text"].strip())
        except Exception as exc:  # 捕获异常后打印出来
            print(exc)
