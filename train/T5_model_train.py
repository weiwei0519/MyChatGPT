# coding=utf-8
# T5 (Text-To-Text transfer transformer) model train
"""
# Project Name: MyGPT
# File Name: T5_model_train
# Author: NSNP577
# Creation at 2023/3/1 9:48
# IDE: PyCharm
# Describe: 
"""

import torch
from torch.utils.data import DataLoader, Dataset
import os
import time
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import Trainer, TrainingArguments, get_linear_schedule_with_warmup
from torch_optimizer import Adafactor
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
from utils.gpu_track import MemTracker
import inspect
import logging

# rich: for a better display on terminal
from rich.table import Column, Table
from rich import box
from rich.console import Console

# 做一些相关的配置(打印显示；GPU设置)
# define a rich console logger
console = Console(record=True)
logging.basicConfig(level=logging.INFO)


def train(epoch, tokenizer, model, device, loader, optimizer):
    """
    用于训练的方法
    Function to be called for training with the parameters passed from main function
    """
    # 追踪GPU Mem的消耗情况。
    frame = inspect.currentframe()  # define a frame to track
    gpu_tracker = MemTracker(frame)

    model.train()
    time1 = time.time()
    gpu_tracker.track()
    for _, data in enumerate(loader, 0):
        y = data["target_ids"].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()  # target, for second to end.e.g."好吗？"
        # releted to pad_token and loss. for detail, check here: https://github.com/Shivanandroy/T5-Finetuning-PyTorch/issues/3
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data["source_ids"].to(device, dtype=torch.long)  # input. e.g. "how are you?"
        mask = data["source_mask"].to(device, dtype=torch.long)

        gpu_tracker.track()

        outputs = model(input_ids=ids, attention_mask=mask, labels=y, )
        loss = outputs[0]
        # 每100步打印日志
        if _ % 1 == 0 and _ != 0:
            time2 = time.time()
            print(_, "epoch:" + str(epoch) + "-loss:" + str(loss) + ";each step's time spent:" + str(
                float(time2 - time1) / float(_ + 0.0001)))
            # training_logger.add_row(str(epoch), str(_), str(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate(epoch, tokenizer, model, device, loader, max_length):
    """
    用于验证的方法：输入用于验证的数据，返回模型预测的结果和正确的标签
    Function to evaluate model for predictions

    """
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype=torch.long)
            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=max_length,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
            )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                     generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]
            if _ % 1000 == 0:
                logging.debug(f'Completed {_}')

            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals


def compute_metrics2(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = (labels == preds).sum() / len(labels)
    return {
        'accuracy': acc,
    }


def model_train(
        tokenizer, model, dataset, batch_size, epochs, learning_rate, device,
        model_dir="./models/CompanyModel0.1-TTT-Chinese",
        log_dir='./logs/TTT-train/',
        datasets_dir='./datasets/company_datasets/',
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
        prediction_loss_only=True,  # 只计算loss不计算evaluation
        gradient_accumulation_steps=256 / batch_size,
        # 显存重计算是典型的用时间换空间，比如我们希望跑256的大点的batch，不希望跑32这样的小batch，
        # 因为觉得小batch不稳定，会影响模型效果，但是gpu显存又无法放下256的batchsize的数据，
        # 此时我们就可以进行显存重计算，将这个参数设置为256/32=8即可。
        # 用torch实现就是forward，计算loss 8次，然后再optimizer.step()
    )

    optimizer = Adafactor(
        model.parameters(),
        lr=learning_rate,
        eps2=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False
    )

    # 学习率变化策略
    total_steps = 0
    if len(dataset) % batch_size == 0:
        total_steps = (len(dataset) // batch_size) * epochs
    else:
        total_steps = (len(dataset) // batch_size + 1) * epochs
    warm_up_ratio = 0.1  # 定义要预热的step
    lr_scheduler = get_linear_schedule_with_warmup(optimizer,
                                                   num_warmup_steps=warm_up_ratio * total_steps,
                                                   num_training_steps=total_steps,
                                                   )
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained 需要训练的模型
        tokenizer=tokenizer,
        args=training_args,  # training arguments, defined above 训练参数
        train_dataset=dataset,  # training dataset 训练集
        eval_dataset=dataset,  # evaluation dataset 测试集
        optimizers=(optimizer, lr_scheduler),  # 自定义优化器
        compute_metrics=compute_metrics  # 计算指标方法
    )

    # print(torch.cuda.memory_summary())

    trainer.train()


# to display dataframe in ASCII format
def display_df(df):
    """display dataframe in ASCII format"""

    # console = Console()
    table = Table(
        Column("source_text", justify="center"),
        Column("target_text", justify="center"),
        title="Sample Data",
        pad_edge=False,
        box=box.ASCII,
    )

    for i, row in enumerate(df.values.tolist()):
        table.add_row(row[0], row[1])

    logging.debug(table)  # TODO TODO TODO
