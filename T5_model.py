# coding=utf-8
# T5 (Text-To-Text transfer transformer) model definition
"""
# Project Name: MyGPT
# File Name: T5_model
# Author: NSNP577
# Creation at 2023/3/1 9:35
# IDE: PyCharm
# Describe: 
"""

import torch
from utils.dataset_util import InputOutputDataset, preprocess, postprocess
from torch.utils.data import DataLoader
import logging
import os
import random
import time
from tqdm import tqdm
import io
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from train.T5_model_train import model_train
from utils.gpu_track import MemTracker
import inspect

# device : GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
logging.basicConfig(level=logging.INFO)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # 防止GPU内存溢出
# 追踪GPU Mem的消耗情况。
frame = inspect.currentframe()  # define a frame to track
gpu_tracker = MemTracker(frame)

# 训练参数
batch_size = 4
epochs = 500
learning_rate = 1e-5  # 学习率
context_length = 512
action = 'train'  # train 训练   validate 测试  prod  生产运行
pretrained_model_dir = "./models/ChatYuan-large-v1/"
model_output_dir = "./models/CompanyModel0.1-TTT-Chinese/"


# 但这样有时可能会出现问题，例如模型陷入一个循环，不断生成同一个单词。
# 为了避免这种情况， GPT-2 设置了一个 top-k 参数，这样模型就会从概率前 k 大的单词中随机选取一个单词，作为下一个单词。
def select_top_k(predictions, k=10):
    predicted_tokens = random.choice(
        predictions[0, -1, :].sort(descending=True)[1][:10]).item()
    return predicted_tokens


'''
GPT-2 核心思想
(translate to french, english text, french text)
(answer the question, document, question, answer)
model类是目前在库中提供的8个模型架构的PyTorch模型(torch.nn.Modules)，例如BertModel
configuration类，它存储构建模型所需的所有参数，例如BertConfig。您不必总是自己实例化这些配置，特别是如果您使用的是未经任何修改的预训练的模型，创建模型将自动负责实例化配置(它是模型的一部分)
tokenizer类，它存储每个模型的词汇表，并在要输送到模型的词汇嵌入索引列表中提供用于编码/解码字符串的方法，例如BertTokenizer
from_pretraining()允许您从一个预训练版本实例化一个模型/配置/tokenizer
save_pretraining()允许您在本地保存模型/配置/tokenizer
'''


def train():
    # gpu_tracker.track()
    # 初始化预训练模型
    tokenizer = T5Tokenizer.from_pretrained(pretrained_model_dir)
    config = T5Config.from_pretrained(pretrained_model_dir)
    model = T5ForConditionalGeneration(config)
    model.to(device)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"model size: {model_size / 1000 ** 2:.1f}M parameters")
    # gpu_tracker.track()

    # 数据集
    file = './datasets/company_datasets/text_to_text_dataset.txt'
    lines = io.open(file, encoding='UTF-8').read().strip().split('\n')
    texts_pairs = [[w for w in l.split('：')] for l in lines]
    source_texts, target_texts = zip(*texts_pairs)
    src_max_len = max([len(text) for text in source_texts])
    tgt_max_len = max([len(text) for text in target_texts])
    train_set = InputOutputDataset(tokenizer=tokenizer,
                                   source_texts=source_texts,
                                   target_texts=target_texts,
                                   source_len=src_max_len,
                                   target_len=tgt_max_len,
                                   )
    print('dataset''s shape = {0}, {1}'.format(train_set.source_shape, train_set.target_shape))

    # 开始模型训练
    pre = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 定义优化器

    model_train(
        tokenizer=tokenizer,
        model=model,
        dataset=train_set,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        model_dir=model_output_dir,
    )
    # trainer.evaluate()
    print('训练时间：', time.time() - pre)


def infer_answer(model, tokenizer, text, sample=True, top_p=1, temperature=0.7):
    '''sample：是否抽样。生成任务，可以设置为True;
    top_p：0-1之间，生成的内容越多样'''
    text = preprocess(text)
    encoding = tokenizer(text=[text],
                         truncation=True,
                         pad_to_max_length=True,
                         padding='max_length',
                         max_length=context_length,
                         return_tensors="pt").to(device)

    if not sample:
        out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False,
                             max_new_tokens=512,
                             num_beams=1, length_penalty=0.6)
    else:
        out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False,
                             max_new_tokens=512,
                             do_sample=True, top_p=top_p, temperature=temperature,
                             no_repeat_ngram_size=3)
    out_text = tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)
    return postprocess(out_text[0])


# 模型测试
def validate(model, tokenizer, input_text):
    total_predicted_text = infer_answer(input_text)

    return "".join(total_predicted_text.split())


if __name__ == '__main__':
    if action == 'train':
        train()
    elif action == 'validate':
        # 加载Company预训练模型
        tokenizer = T5Tokenizer.from_pretrained(model_output_dir)
        config = T5Config.from_pretrained(model_output_dir)
        model = T5ForConditionalGeneration.from_pretrained(model_output_dir)
        model.to(device)
        cont = True
        while cont:
            input_text = str(input("请输入/Please input： "))

            if input_text == "exit":
                cont = False
            else:
                output_text = infer_answer(model=model, tokenizer=tokenizer, text=input_text)
                print(output_text)
