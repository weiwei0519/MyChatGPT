# coding=UTF-8
# 代码补全GPT

'''
@File: MyCodeGPT_train
@Author: WeiWei
@Time: 2023/2/23
@Email: weiwei_519@outlook.com
@Software: PyCharm
'''

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./code-search-net-tokenizer")