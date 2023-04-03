# coding=UTF-8
# 用训练好的GPT模型进行推理

'''
@File: infer
@Author: WeiWei
@Time: 2023/4/2
@Email: weiwei_519@outlook.com
@Software: PyCharm
'''

import torch
from chatbot_with_gpt.gpt_model import GPT

if __name__ == '__main__':
    device = torch.device('cuda')
    model = GPT().to(device)
    model.load_state_dict(torch.load('GPT2.pt'))

    model.eval()
    # 初始输入是空，每次加上后面的对话信息
    sentence = ''
    while True:
        temp_sentence = input("我:")
        sentence += (temp_sentence + '\t')
        if len(sentence) > 200:
            # 由于该模型输入最大长度为300，避免长度超出限制长度过长需要进行裁剪
            t_index = sentence.find('\t')
            sentence = sentence[t_index + 1:]
        print("机器人:", model.answer(sentence))
