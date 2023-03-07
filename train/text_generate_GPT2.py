# coding=UTF-8
# Pytorch——GPT-2 预训练模型及文本生成

'''
@File: GPT2_model
@Author: WeiWei
@Time: 2023/2/18
@Email: weiwei_519@outlook.com
@Software: PyCharm
'''

"""
GPT-2 核心思想
(translate to french, english text, french text)
(answer the question, document, question, answer)
"""

import random
import torch
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel
import logging
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from torch.autograd import Variable
import time
from tqdm import tqdm

# device : GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 设置训练参数
batch_size = 4
learning_rate = 1e-3
epochs = 100
action = 'train'  # train 训练   validate 测试  prod  生产运行


# 但这样有时可能会出现问题，例如模型陷入一个循环，不断生成同一个单词。
# 为了避免这种情况， GPT-2 设置了一个 top-k 参数，这样模型就会从概率前 k 大的单词中随机选取一个单词，作为下一个单词。
def select_top_k(predictions, k=10):
    predicted_tokens = random.choice(
        predictions[0, -1, :].sort(descending=True)[1][:10]).item()
    return predicted_tokens


logging.basicConfig(level=logging.INFO)

# 载入预训练模型的分词器
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# 使用 GPT2Tokenizer 对输入进行编码
text = "Yesterday, a man named Jack said he saw an alien,"
text_tokens = tokenizer.encode(text)
tokens_tensor = torch.tensor([text_tokens])
print(tokens_tensor)

# 读取 GPT-2 预训练模型
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

total_predicted_text = text
n = 100  # 预测过程的循环次数
for _ in range(n):
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

    predicted_tokens = select_top_k(predictions, k=10)
    predicted_text = tokenizer.decode(predicted_tokens)
    total_predicted_text += predicted_text

    if '<|endoftext|>' in total_predicted_text:
        # 如果出现文本结束标志，就结束文本生成
        break

    text_tokens += [predicted_tokens]
    tokens_tensor = torch.tensor([text_tokens])

print(total_predicted_text)

# 通过新的文本集训练进行微调
with open('./datasets/generative_datasets/doc_dataset.txt', 'r', encoding='UTF-8') as f:
    dataset = f.read()
print('datasets length = {0}'.format(len(dataset)))

# 预处理训练集，将训练集编码、分段
indexed_text = tokenizer.encode(dataset)
del dataset

dataset_cut = []
for i in range(len(indexed_text) // 512):
    # 将字符串分段成长度为 512
    dataset_cut.append(indexed_text[i * 512:i * 512 + 512])
del indexed_text

dataset_tensor = torch.tensor(dataset_cut)
print('datasets shape = {0}'.format(dataset_tensor.shape))

# 构建数据集和数据迭代器，设定 batch_size 大小为 2
train_set = TensorDataset(dataset_tensor,
                          dataset_tensor)  # 标签与样本数据相同
train_loader = DataLoader(dataset=train_set,
                          batch_size=batch_size,
                          shuffle=False)

# 开始模型训练
pre = time.time()

model.to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # 定义优化器

for i in range(epochs):
    total_loss = 0
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        data, target = Variable(data).to(device), Variable(
            target).to(device)

        optimizer.zero_grad()

        loss, logits, _ = model(data, labels=target)

        total_loss += loss

        loss.backward()
        optimizer.step()

        if batch_idx == len(train_loader) - 1:
            # 在每个 Epoch 的最后输出一下结果
            print('epoch: {0}   average loss: {1}'.format(i, total_loss / len(train_loader)))

print('训练时间：', time.time() - pre)

# 模型测试
text = "友童乐齿医疗保险合同内容变更时"  # 这里也可以输入不同的英文文本
text_tokens = tokenizer.encode(text)
tokens_tensor = torch.tensor([text_tokens])

model.eval()
total_predicted_text = text

# 使训练后的模型进行 500 次预测
for _ in range(500):
    tokens_tensor = tokens_tensor.to('cuda')

    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

    predicted_tokens = select_top_k(predictions, k=10)

    predicted_text = tokenizer.decode(text_tokens + [predicted_tokens])
    total_predicted_text += tokenizer.decode(predicted_tokens)
    if '<|endoftext|>' in total_predicted_text:
        # 如果出现文本结束标志，就结束文本生成
        break

    text_tokens += [predicted_tokens]

    if len(text_tokens) > 1023:
        # 模型最长输入长度为1024，如果长度过长则截断
        text_tokens = text_tokens[-1023:]

    tokens_tensor = torch.tensor([text_tokens])

print(total_predicted_text)
