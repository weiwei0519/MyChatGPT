# coding=utf-8
# 使用transformer训练自己的模型
"""
# Project Name: MyGPT
# File Name: MyGPT_train
# Author: NSNP577
# Creation at 2023/2/21 13:27
# IDE: PyCharm
# Describe: 
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
import logging
import os
import random
import time
from tqdm import tqdm
from transformers import Trainer, TrainingArguments
from transformers import GPT2LMHeadModel, TextGenerationPipeline, GPT2Tokenizer, BertTokenizer
from transformers import WEIGHTS_NAME, CONFIG_NAME, GPT2Config
from tokenizers import Tokenizer
from tokenizers.models import BPE
import docx2txt

# device : GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
logging.basicConfig(level=logging.INFO)

# 训练参数
batch_size = 256
epochs = 5000
learning_rate = 1e-5  # 学习率
max_len = 512
action = 'validate'  # train 训练   validate 测试  prod  生产运行
model_dir = "./models/CompanyModel0.1-GPT2-Chinese/"


# 但这样有时可能会出现问题，例如模型陷入一个循环，不断生成同一个单词。
# 为了避免这种情况， GPT-2 设置了一个 top-k 参数，这样模型就会从概率前 k 大的单词中随机选取一个单词，作为下一个单词。
def select_top_k(predictions, k=10):
    predicted_tokens = random.choice(
        predictions[0, -1, :].sort(descending=True)[1][:10]).item()
    return predicted_tokens

'''
model类是目前在库中提供的8个模型架构的PyTorch模型(torch.nn.Modules)，例如BertModel
configuration类，它存储构建模型所需的所有参数，例如BertConfig。您不必总是自己实例化这些配置，特别是如果您使用的是未经任何修改的预训练的模型，创建模型将自动负责实例化配置(它是模型的一部分)
tokenizer类，它存储每个模型的词汇表，并在要输送到模型的词汇嵌入索引列表中提供用于编码/解码字符串的方法，例如BertTokenizer
from_pretraining()允许您从一个预训练版本实例化一个模型/配置/tokenizer
save_pretraining()允许您在本地保存模型/配置/tokenizer
'''
'''
config = GPT2Config(
    architectures=["GPT2LMHeadModel"],   # pretrain的时候用来预加载模型
    model_type="GPT2LMHeadModel",        # 定义模型类型，导出给`AutoConfig`用，如果要上传到hub请必填
    tokenizer_class="GPT2Tokenizer",       # 定义tokenizer类型，导出给`AutoTokenizer`用，如果要上传到hub请必填
    vocab_size=8021,
    n_positions=1024,
    n_ctx=1024,
    n_embd=768,
    n_layer=6,
    n_head=6,
    pad_token_id=tokenizer.pad_token_id,   # 前面构建的tokenizer的 PAD ID
    task_specific_params={
        "text-generation": {
            "do_sample": True,
            "max_length": 120
        }
    }
)
'''
def train():
    # 初始化transformer模型(空模型)
    # tokenizer = Tokenizer.from_pretrained('./models/CompanyModel0.1-GPT2-Chinese')
    tokenizer = Tokenizer(BPE())
    tokenizer.model = BPE.from_file('./models/CompanyModel0.1-GPT2-Chinese/vocab.json',
                                    './models/CompanyModel0.1-GPT2-Chinese/merges.txt')
    print('initial pre-trained Tokenizer')
    config = GPT2Config(
        architectures=["GPT2LMHeadModel"],   # pretrain的时候用来预加载模型
        model_type="GPT2LMHeadModel",        # 定义模型类型，导出给`AutoConfig`用，如果要上传到hub请必填
        tokenizer_class="Tokenizer",       # 定义tokenizer类型，导出给`AutoTokenizer`用，如果要上传到hub请必填
        vocab_size=8021,
        n_positions=1024,
        n_ctx=1024,
        n_embd=768,
        n_layer=6,
        n_head=6,
        pad_token_id=0,   # 前面构建的tokenizer的 PAD ID
        task_specific_params={
            "text-generation": {
                "do_sample": True,
                "max_length": 120
            }
        }
    )
    print('initial GPT2 Config')
    model = GPT2LMHeadModel(config=config)
    print('initial GPT2 Model, num parameters: = {0}'.format(model.num_parameters()))
    # model = GPT2LMHeadModel.from_pretrained('./models/gpt2-chinese-cluecorpussmall')
    # 加载中文GPT2模型
    # tokenizer = GPT2Tokenizer.from_pretrained('./models/Wenzhong2.0-GPT2-3.5B-chinese')
    # model = GPT2LMHeadModel.from_pretrained('./models/Wenzhong2.0-GPT2-3.5B-chinese')
    text_generator = TextGenerationPipeline(model, tokenizer)
    print('initial GPT2 text generator')
    model.to(device)

    # 数据集
    # 通过新的文本集训练进行微调
    # with open('./datasets/generative_datasets/romeo_and_juliet.txt', 'r', encoding='UTF-8') as f:
    #     text = f.read()
    # print('datasets length = {0}'.format(len(text)))

    doc_path = './datasets/document/'
    files = os.listdir(doc_path)
    doc_texts = []
    for file in files:
        # f = open(doc_path + file, 'r', encoding='UTF-8')
        text = docx2txt.process(doc_path + file)
        doc_texts.append(text.replace("\n\n", "\n"))

    # 预处理训练集，将训练集编码、分段
    dataset = []
    # 截取solution
    # for i in range(len(text) // max_len):
    #     # 将字符串分段成长度为max_len为单位
    #     dataset.extend(tokenizer.encode(text=text[i * max_len:(i + 1) * max_len]))
    # del text
    # 不截取solution
    for i in range(len(doc_texts)):
        # 将字符串分段成长度为max_len为单位
        dataset.extend(tokenizer.encode(doc_texts[i]).ids)
    del text

    dataset_tensor = torch.tensor(dataset)
    print('datasets shape = {0}'.format(dataset_tensor.shape))

    # 构建数据集和数据迭代器，设定 batch_size 大小为 2
    train_set = TensorDataset(dataset_tensor,
                              dataset_tensor)  # 标签与样本数据相同
    train_loader = DataLoader(dataset=train_set,
                              batch_size=batch_size,
                              shuffle=False)

    # 开始模型训练
    pre = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 定义优化器

    for i in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = Variable(data).to(device), Variable(target).to(device)
            # 每轮epoch之前，先清零梯度
            optimizer.zero_grad()
            # 计算输出和loss
            # loss, logits, _ = model(data, labels=target)
            output = model(data, labels=target)
            loss = output.loss
            logits = output.logits
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            if batch_idx == len(train_loader) - 1:
                # 在每个 Epoch 的最后输出一下结果
                print('epoch: {0}   average loss: {1}'.format(i, total_loss / len(train_loader)))

    #保存经过微调预训练模型的权重、配置和词汇表
    model_to_save = model.module if hasattr(model, 'module') else model
    # 使用预定义的名称保存，则可以使用`from_pretrained`加载
    output_model_file = os.path.join(model_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(model_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    # tokenizer.save_vocabulary(model_dir)
    print('训练时间：', time.time() - pre)

# 模型测试
def validate(text):
    # 加载预训练模型：
    model = GPT2LMHeadModel.from_pretrained('./models/CompanyModel0.1-GPT2-Chinese')
    tokenizer = Tokenizer(BPE())
    tokenizer.model = BPE.from_file('./models/CompanyModel0.1-GPT2-Chinese/vocab.json',
                                    './models/CompanyModel0.1-GPT2-Chinese/merges.txt')
    text_tokens = tokenizer.encode(text).ids
    tokens_tensor = torch.tensor([text_tokens])
    model.to(device)
    model.eval()
    total_predicted_text = text

    # 使训练后的模型进行 500 次预测
    for _ in range(500):
        tokens_tensor = tokens_tensor.to(device)

        with torch.no_grad():
            outputs = model(tokens_tensor)
            predictions = outputs[0]

        predicted_tokens = select_top_k(predictions, k=10)

        predicted_text = tokenizer.decode(text_tokens + [predicted_tokens])
        total_predicted_text += tokenizer.decode([predicted_tokens])
        if '<|endoftext|>' in total_predicted_text:
            # 如果出现文本结束标志，就结束文本生成
            break

        text_tokens += [predicted_tokens]

        if len(text_tokens) > 1023:
            # 模型最长输入长度为1024，如果长度过长则截断
            text_tokens = text_tokens[-1023:]

        tokens_tensor = torch.tensor([text_tokens])

    print(total_predicted_text)

# X = torch.zeros((26, 26), dtype=torch.float32).to(device=device)
# labels = []
# for i in range(26):
#     labels.append((i + 1) % 26)
#     X[i][i] = 1.
# labels = torch.tensor(labels)
# dataset = Dataset.from_dict({'x': X, 'labels': labels})
#
#
# # 残差网络
# class RN(nn.Module):
#     def __init__(self):
#         super(RN, self).__init__()
#         self.linear_stack = nn.Sequential(
#             nn.Linear(26, 64),
#             nn.Hardsigmoid(),
#             nn.Linear(64, 26),
#             nn.Hardsigmoid(),
#         )
#
#         self.linear_stack_2 = nn.Sequential(
#             nn.Linear(26, 64),
#             nn.Hardsigmoid(),
#             nn.Linear(64, 64),
#             nn.Hardsigmoid(),
#         )
#
#         self.output_layer = nn.Linear(64, 26)
#
#         self.loss_f = nn.CrossEntropyLoss()
#
#     def forward(self, x, labels, mode='train'):
#         y = self.linear_stack(x)
#         # 残差
#         y = y + x
#         y = self.linear_stack_2(y)
#         y = self.output_layer(y)
#
#         if mode == 'train':
#             return {
#                 'loss': self.loss_f(y, labels),
#                 'predictions': y
#             }
#
#         return y
#
#
# # 生成模型实例
# model = RN().to(device=device)
#
#
# def compute_metrics(pred):
#     labels = pred.label_ids
#     preds = pred.predictions.argmax(-1)
#     acc = (labels == preds).sum() / len(labels)
#     return {
#         'accuracy': acc,
#     }
#
#
# training_args = TrainingArguments(
#     output_dir='./models/CompanyModel0.1-GPT2-Chinese',  # output directory 结果输出地址
#     num_train_epochs=epochs,  # total # of training epochs 训练总批次
#     per_device_train_batch_size=batch_size,  # batch size per device during training 训练批大小
#     per_device_eval_batch_size=batch_size,  # batch size for evaluation 评估批大小
#     logging_dir='./logs/rn_log',  # directory for storing logs 日志存储位置
#     learning_rate=learning_rate,  # 学习率
#     save_steps=False,  # 不保存检查点
# )
#
# trainer = Trainer(
#     model=model,  # the instantiated 🤗 Transformers model to be trained 需要训练的模型
#     args=training_args,  # training arguments, defined above 训练参数
#     train_dataset=train_set,  # training dataset 训练集
#     eval_dataset=train_set,  # evaluation dataset 测试集
#     compute_metrics=compute_metrics  # 计算指标方法
# )
#
# trainer.train()
# trainer.evaluate()

if __name__ == '__main__':
    if action == 'train':
        train()
    elif action == 'validate':
        cont = True
        while cont:
            input_text = str(input("请输入/Please input： "))

            if input_text == "exit":
                cont = False
            else:
                validate(input_text)
