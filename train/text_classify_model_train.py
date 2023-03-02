# coding=UTF-8
# reward model——奖励模型的训练

'''
@File: reward_model_train
@Author: WeiWei
@Time: 2023/2/26
@Email: weiwei_519@outlook.com
@Software: PyCharm
'''

import torch
from torch.utils.data import DataLoader, Dataset
import os
import time
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import Trainer, TrainingArguments
from utils.gpu_track import MemTracker
import inspect
import logging

# device : GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
logging.basicConfig(level=logging.INFO)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = (labels == preds).sum() / len(labels)
    return {
        'accuracy': acc,
    }

def model_train(
        tokenizer, model, train_dataset, val_dataset, batch_size, epochs, learning_rate, device,
        model_dir="./models/Company-RewardModel0.1-Chinese/",
        log_dir='./logs/reward_model_train/',
        datasets_dir='./datasets/IMDB_movie/aclImdb/',
):
    # 参数参考这篇文章：https://zhuanlan.zhihu.com/p/363670628
    training_args = TrainingArguments(
        output_dir=model_dir,  # output directory 结果输出地址
        num_train_epochs=epochs,  # total # of training epochs 训练总批次
        per_device_train_batch_size=batch_size,  # batch size per device during training 训练批大小
        per_device_eval_batch_size=batch_size,  # batch size for evaluation 评估批大小
        evaluation_strategy="steps",  # Evaluation is done at the end of each epoch. or 10 steps
        logging_dir=log_dir,  # directory for storing logs 日志存储位置
        logging_strategy='epoch',
        learning_rate=learning_rate,  # 学习率
        save_strategy='epoch',  # 不保存检查点
        save_total_limit=1,  # 只保留一个checkpoint
        overwrite_output_dir=True,  # 覆盖之前写的模型输出文件
        gradient_accumulation_steps=256 / batch_size,
        # 显存重计算是典型的用时间换空间，比如我们希望跑256的大点的batch，不希望跑32这样的小batch，
        # 因为觉得小batch不稳定，会影响模型效果，但是gpu显存又无法放下256的batchsize的数据，
        # 此时我们就可以进行显存重计算，将这个参数设置为256/32=8即可。
        # 用torch实现就是forward，计算loss 8次，然后再optimizer.step()
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained 需要训练的模型
        tokenizer=tokenizer,
        args=training_args,  # training arguments, defined above 训练参数
        train_dataset=train_dataset,  # training dataset 训练集
        eval_dataset=val_dataset,  # evaluation dataset 测试集
        compute_metrics=compute_metrics  # 计算指标方法
    )

    trainer.train()
