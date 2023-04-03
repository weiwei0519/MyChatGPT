# coding=UTF-8
# 生成词典

'''
@File: get_vocab
@Author: WeiWei
@Time: 2023/4/2
@Email: weiwei_519@outlook.com
@Software: PyCharm
'''

import json


def get_dict(datas):
    word_count = {}
    for data in datas:
        data = data.strip().replace('\t', '')
        for word in data:
            word_count.setdefault(word, 0)
            word_count[word] += 1
    word2id = {"<pad>": 0, "<unk>": 1, "<sep>": 2}

    temp = {word: i + len(word2id) for i, word in enumerate(word_count.keys())}
    word2id.update(temp)
    id2word = list(word2id.keys())
    return word2id, id2word


if __name__ == '__main__':
    with open('dataset.txt', 'r', encoding='utf-8') as f:
        datas = f.readlines()
    word2id, id2word = get_dict(datas)

    dict_datas = {"word2id": word2id, "id2word": id2word}

    json.dump(dict_datas, open('dict_datas.json', 'w', encoding='utf-8'))
