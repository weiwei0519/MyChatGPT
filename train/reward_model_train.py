# !/usr/bin/env python3
# 使用ChatGPT中Reward Model的思路训练一个RM，因为是打分模型，所以使用BERT（而非GPT）模型训练。

"""
# Project Name: MyChatGPT
# File Name: training_logger
# Author: NSNP577
# Creation at 2023/3/2 11:42
# IDE: PyCharm
# Describe:
"""
import os
import io
import time
import argparse
import logging
from functools import partial
from tqdm import tqdm
from rich import print
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, default_data_collator, get_scheduler, GPT2LMHeadModel
from transformers import get_linear_schedule_with_warmup
from transformers import TrainingArguments, DataCollatorWithPadding
from reward_model import RewardModel, RewardModelTrainer
from utils.dataset_util import dataset_process, TextRewardDataset
from utils.training_logger import LoggerDisplayer
from torch_optimizer import Adafactor

# device : GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print(device)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # 防止GPU内存溢出

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="../models/bert-base-chinese", type=str, help="backbone of encoder.")
parser.add_argument("--train_path", default="../datasets/reward_model_dataset/train.tsv", type=str,
                    help="The path of train set.")
parser.add_argument("--val_path", default="../datasets/reward_model_dataset/val.tsv", type=str,
                    help="The path of dev set.")
parser.add_argument("--save_dir", default="../models/CompanyModel0.1-RewardModel-Chinese", type=str, required=False,
                    help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--max_seq_len", default=512, type=int,
                    help="The maximum total input sequence length after tokenization. Sequences longer "
                         "than this will be truncated, sequences shorter will be padded.", )
parser.add_argument("--batch_size", default=2, type=int, help="Batch size per GPU/CPU for training.", )
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--epochs", default=100, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--warmup_ratio", default=0.0, type=float, help="Linear warmup over warmup_ratio * total_steps.")
parser.add_argument("--valid_steps", default=200, type=int, required=False, help="evaluate frequecny.")
parser.add_argument("--logging_steps", default=10, type=int, help="log interval.")
parser.add_argument("--img_log_dir", default='./logs/reward_model_train', type=str, help="Logging image path.")
parser.add_argument("--img_log_name", default='reward_model.log', type=str, help="Logging image file name.")
args = parser.parse_args()

# logger_displayer = LoggerDisplayer(log_path=args.img_log_dir, log_name=args.img_log_name)
logging.basicConfig(level=logging.INFO)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = (labels == preds).sum() / len(labels)
    return {
        'accuracy': acc,
    }


def evaluate_model(model, data_loader):
    """
    在测试集上评估当前模型的训练效果。

    Args:
        model: 当前模型
        data_loader: 测试集的dataloader
    """
    model.eval()
    with torch.no_grad():
        batch_rank_rewards = []
        for batch in data_loader:
            for batch_idx in range(len(batch['input_ids'])):
                rank_texts_count = len(batch['input_ids'][batch_idx])
                rank_rewards = []
                for text_idx in range(rank_texts_count):
                    reward = model(
                        batch['input_ids'][batch_idx][text_idx].unsqueeze(dim=0).to(device),
                        batch['token_type_ids'][batch_idx][text_idx].unsqueeze(dim=0).to(device),
                        batch['attention_mask'][batch_idx][text_idx].unsqueeze(dim=0).to(device),
                        batch['position_ids'][batch_idx][text_idx].unsqueeze(dim=0).to(device),
                    )
                    rank_rewards.append(reward[0])  # (rank_text_num) -> [tensor([0.1696]), tensor([0.3466])]
                batch_rank_rewards.append(
                    rank_rewards)  # (batch, rank_text_num) -> [[tensor([0.1696]), tensor([0.3466])], ...]
    model.train()
    total_ranklist, right_ranklist = 0, 0
    for rank_rewards in batch_rank_rewards:
        rank_rewards = [t.cpu().float() for t in rank_rewards]
        rank_rewards_sorted = sorted(rank_rewards, reverse=True)
        total_ranklist += 1
        if rank_rewards_sorted == rank_rewards:
            right_ranklist += 1
    return right_ranklist / total_ranklist


def train():
    encoder = AutoModel.from_pretrained(args.model)
    model = RewardModel(encoder=encoder)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"model size: {model_size / 1000 ** 2:.1f}M parameters")
    encoder.to(device)
    model.to(device)
    # dataset = load_dataset('text', data_files={'train': args.train_path,
    #                                            'val': args.dev_path})
    # print(dataset)
    # # dataset = dataset_process(dataset=dataset, tokenizer=tokenizer, max_seq_len=args.max_seq_len)
    # convert_func = partial(dataset_process, tokenizer=tokenizer, max_seq_len=args.max_seq_len)
    # dataset = dataset.map(convert_func)  # batched=True采用批量处理的方式，提升performance
    #
    # train_dataset = dataset["train"]
    # eval_dataset = dataset["val"]
    # 加载训练集
    texts = io.open(args.train_path, encoding='UTF-8').read().strip().split('\n')
    train_dataset = TextRewardDataset(texts=texts, tokenizer=tokenizer, max_len=args.max_seq_len)
    # 加载测试集
    texts = io.open(args.val_path, encoding='UTF-8').read().strip().split('\n')
    val_dataset = TextRewardDataset(texts=texts, tokenizer=tokenizer, max_len=args.max_seq_len)
    train_dataloader = DataLoader(train_dataset, shuffle=False, collate_fn=default_data_collator,
                                  batch_size=args.batch_size)
    eval_dataloader = DataLoader(val_dataset, shuffle=False, collate_fn=default_data_collator,
                                 batch_size=args.batch_size)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # 根据训练轮数计算最大训练步数，以便于scheduler动态调整lr
    num_update_steps_per_epoch = len(train_dataloader)
    max_train_steps = args.epochs * num_update_steps_per_epoch
    warm_steps = int(args.warmup_ratio * max_train_steps)
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warm_steps,
        num_training_steps=max_train_steps,
    )

    training_args = TrainingArguments(
        output_dir=args.save_dir,  # output directory 结果输出地址
        num_train_epochs=args.epochs,  # total # of training epochs 训练总批次
        per_device_train_batch_size=args.batch_size,  # batch size per device during training 训练批大小
        per_device_eval_batch_size=args.batch_size,  # batch size for evaluation 评估批大小
        evaluation_strategy="steps",  # Evaluation is done at the end of each epoch. or 10 steps
        logging_dir=args.img_log_dir,  # directory for storing logs 日志存储位置
        logging_strategy='epoch',
        learning_rate=args.learning_rate,  # 学习率
        save_strategy='epoch',  # 不保存检查点
        save_total_limit=1,  # 只保留一个checkpoint
        overwrite_output_dir=True,  # 覆盖之前写的模型输出文件
        prediction_loss_only=True,  # 只计算loss不计算evaluation
        gradient_accumulation_steps=256 / args.batch_size,
        # 显存重计算是典型的用时间换空间，比如我们希望跑256的大点的batch，不希望跑32这样的小batch，
        # 因为觉得小batch不稳定，会影响模型效果，但是gpu显存又无法放下256的batchsize的数据，
        # 此时我们就可以进行显存重计算，将这个参数设置为256/32=8即可。
        # 用torch实现就是forward，计算loss 8次，然后再optimizer.step()
    )

    # call自定义Trainer，重写了compute_loss
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = RewardModelTrainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained 需要训练的模型
        tokenizer=tokenizer,
        args=training_args,  # training arguments, defined above 训练参数
        train_dataset=train_dataset,  # training dataset 训练集
        eval_dataset=val_dataset,  # evaluation dataset 测试集
        optimizers=(optimizer, lr_scheduler),  # 自定义优化器
        data_collator=data_collator,  # 使用动态padding，节省训练内存占用
        compute_metrics=compute_metrics  # 计算指标方法
    )

    # print(torch.cuda.memory_summary())
    trainer.train()

    # loss_list = []
    # tic_train = time.time()
    # global_step, best_acc = 0, 0
    # for epoch in range(1, args.epochs + 1):
    #     print(f"epoch: {epoch}")
    #     for _, batch in enumerate(tqdm(train_dataloader)):
    #         batch_rank_rewards = []
    #         for batch_idx in range(len(batch['input_ids'])):
    #             rank_texts_count = len(batch['input_ids'][batch_idx])
    #             rank_rewards = []
    #             for text_idx in range(rank_texts_count):
    #                 reward = model(
    #                     batch['input_ids'][batch_idx][text_idx].unsqueeze(dim=0).to(device),
    #                     batch['token_type_ids'][batch_idx][text_idx].unsqueeze(dim=0).to(device),
    #                     batch['attention_mask'][batch_idx][text_idx].unsqueeze(dim=0).to(device),
    #                     batch['position_ids'][batch_idx][text_idx].unsqueeze(dim=0).to(device),
    #                 )
    #                 rank_rewards.append(reward[0])  # (rank_text_num) -> [tensor([0.1696]), tensor([0.3466])]
    #             batch_rank_rewards.append(
    #                 rank_rewards)  # (batch, rank_text_num) -> [[tensor([0.1696]), tensor([0.3466])], ...]
    #         loss = compute_rank_list_loss(batch_rank_rewards)
    #         loss.backward()
    #         optimizer.step()
    #         lr_scheduler.step()
    #         optimizer.zero_grad()
    #         loss_list.append(float(loss.cpu().detach()))
    #
    #         global_step += 1
    #         if global_step % args.logging_steps == 0:
    #             time_diff = time.time() - tic_train
    #             loss_avg = sum(loss_list) / len(loss_list)
    #             logger_displayer.add_scalar('train/train_loss', loss_avg, global_step)
    #             print("global step %d, epoch: %d, loss: %.5f, speed: %.2f step/s"
    #                   % (global_step, epoch, loss_avg, args.logging_steps / time_diff))
    #             tic_train = time.time()
    #
    #         if global_step % args.valid_steps == 0:
    #             cur_save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
    #             if not os.path.exists(cur_save_dir):
    #                 os.makedirs(cur_save_dir)
    #             torch.save(model, os.path.join(cur_save_dir, 'model.pt'))
    #             tokenizer.save_pretrained(cur_save_dir)
    #             acc = evaluate_model(model, eval_dataloader)
    #             logger_displayer.add_scalar('eval/accuracy', acc, global_step)
    #             logger_displayer.record()
    #             print("Evaluation acc: %.5f" % (acc))
    #             if acc > best_acc:
    #                 print(
    #                     f"best F1 performence has been updated: {best_acc:.5f} --> {acc:.5f}"
    #                 )
    #                 best_acc = acc
    #                 cur_save_dir = os.path.join(args.save_dir, "model_best")
    #                 if not os.path.exists(cur_save_dir):
    #                     os.makedirs(cur_save_dir)
    #                 torch.save(model, os.path.join(cur_save_dir, 'model.pt'))
    #                 tokenizer.save_pretrained(cur_save_dir)
    #             tic_train = time.time()


if __name__ == '__main__':
    train()
