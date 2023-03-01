# coding=UTF-8
# GPT-2 模型在数据预处理时使用了字节对编码（Byte Pair Encoding，简称 BPE）方法，
# BPE 是一种能够解决未登录词问题，并减小词典大小的方法。它综合利用了单词层面编码和字符层面编码的优势

'''
@File: BPE
@Author: WeiWei
@Time: 2023/2/18
@Email: weiwei_519@outlook.com
@Software: PyCharm
'''

import re
import collections

# BPE编码
# example：aaabdaaabac ——> ZabdZabac(Z=aa) ——> ZYdZYac(Y=ab & Z=aa) ——> XdXac(X=ZY & Y=ab & Z=aa)
def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq  # 计算字节对出现频率
    return pairs


# BPE解码
# 编码的逆向过程。
def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))  # 将字节对中可解释为正则运算符的字符转义
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')  # 将要合并的字节对前后只能为空白字符
    for word in v_in:
        w_out = p.sub(''.join(pair), word)  # 合并符合条件的字节对
        v_out[w_out] = v_in[word]
    return v_out


vocab = {'l o w </w>': 5, 'l o w e r </w>': 2,
         'n e w e s t </w>': 6, 'w i d e s t </w>': 3}
num_merges = 10
for i in range(num_merges):
    pairs = get_stats(vocab)
    best = max(pairs, key=pairs.get)  # 选择频率最大的字节对
    vocab = merge_vocab(best, vocab)
    print(best)
