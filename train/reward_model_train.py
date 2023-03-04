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
from functools import partial
from tqdm import tqdm
from rich import print
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, default_data_collator, get_scheduler, GPT2LMHeadModel
from reward_model import RewardModel, compute_rank_list_loss
from utils.dataset_util import dataset_process, TextRewardDataset
from utils.training_logger import LoggerDisplayer

# device : GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

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
parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.", )
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--num_train_epochs", default=10, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--warmup_ratio", default=0.0, type=float, help="Linear warmup over warmup_ratio * total_steps.")
parser.add_argument("--valid_steps", default=200, type=int, required=False, help="evaluate frequecny.")
parser.add_argument("--logging_steps", default=10, type=int, help="log interval.")
parser.add_argument("--img_log_dir", default='./logs/reward_model_train', type=str, help="Logging image path.")
parser.add_argument("--img_log_name", default='reward_model.log', type=str, help="Logging image file name.")
args = parser.parse_args()

logger_displayer = LoggerDisplayer(log_path=args.img_log_dir, log_name=args.img_log_name)


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
    model.to(device)

    # 根据训练轮数计算最大训练步数，以便于scheduler动态调整lr
    num_update_steps_per_epoch = len(train_dataloader)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    warm_steps = int(args.warmup_ratio * max_train_steps)
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warm_steps,
        num_training_steps=max_train_steps,
    )

    loss_list = []
    tic_train = time.time()
    global_step, best_acc = 0, 0
    for epoch in range(1, args.num_train_epochs + 1):
        print(f"epoch: {epoch}")
        for _, batch in enumerate(tqdm(train_dataloader)):
            batch_rank_rewards = []
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
            loss = compute_rank_list_loss(batch_rank_rewards, device=device)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            loss_list.append(float(loss.cpu().detach()))

            global_step += 1
            if global_step % args.logging_steps == 0:
                time_diff = time.time() - tic_train
                loss_avg = sum(loss_list) / len(loss_list)
                logger_displayer.add_scalar('train/train_loss', loss_avg, global_step)
                print("global step %d, epoch: %d, loss: %.5f, speed: %.2f step/s"
                      % (global_step, epoch, loss_avg, args.logging_steps / time_diff))
                tic_train = time.time()

            if global_step % args.valid_steps == 0:
                cur_save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
                if not os.path.exists(cur_save_dir):
                    os.makedirs(cur_save_dir)
                torch.save(model, os.path.join(cur_save_dir, 'model.pt'))
                tokenizer.save_pretrained(cur_save_dir)
                acc = evaluate_model(model, eval_dataloader)
                logger_displayer.add_scalar('eval/accuracy', acc, global_step)
                logger_displayer.record()
                print("Evaluation acc: %.5f" % (acc))
                if acc > best_acc:
                    print(
                        f"best F1 performence has been updated: {best_acc:.5f} --> {acc:.5f}"
                    )
                    best_acc = acc
                    cur_save_dir = os.path.join(args.save_dir, "model_best")
                    if not os.path.exists(cur_save_dir):
                        os.makedirs(cur_save_dir)
                    torch.save(model, os.path.join(cur_save_dir, 'model.pt'))
                    tokenizer.save_pretrained(cur_save_dir)
                tic_train = time.time()


if __name__ == '__main__':
    train()
