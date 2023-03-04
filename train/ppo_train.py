# coding=utf-8
# uses Proximal Policy Optimization to optimise language models.
"""
# Project Name: MyChatGPT
# File Name: ppo_train
# Author: NSNP577
# Creation at 2023/3/4 19:25
# IDE: PyCharm
# Describe: 
"""

import os
import time
import random

import torch
from rich import print
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from GPT2_model import GPT2HeadWithValueModel
from model.ppo_model import PPOModel
from utils.training_logger import LoggerWriter

writer = LoggerWriter(log_path='../logs/ppo_train', log_name='PPO-train-Zh')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe_device = 0 if torch.cuda.is_available() else -1

# model config
config = {
    "model_name": '../models/gpt2-chinese-cluecorpussmall',
    "steps": 20000,
    "batch_size": 128,
    "forward_batch_size": 16,
    "ppo_epochs": 4,
    "lr": 1.41e-5,
    "init_kl_coef": 0.2,
    "target": 6,
    "horizon": 10000,
    "gamma": 1,
    "lam": 0.95,
    "cliprange": .2,
    "cliprange_value": .2,
    "vf_coef": .1,
    "gen_len": 128,
    "save_freq": 5,
    'save_dir': '../models/CompanyModel0.1-PPO-Chinese'
}
# 分类模型
sentiment_model_dir = '../models/roberta-base-finetuned-jd-binary-chinese'

# prompt池
prompts = [
    '刚收到货，感觉',
    '这部电影很',
    '说实话，真的很',
    '这次购物总的来说体验很'
]

# 情感分类模型
senti_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_dir)
senti_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_dir)
sentiment_pipe = pipeline('sentiment-analysis', model=senti_model, tokenizer=senti_tokenizer, device=pipe_device)

# 文本生成模型
gpt2_model = GPT2HeadWithValueModel.from_pretrained(config['model_name'])
gpt2_model_ref = GPT2HeadWithValueModel.from_pretrained(config['model_name'])
gpt2_tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
gpt2_tokenizer.eos_token = gpt2_tokenizer.pad_token
gpt2_model.to(device)
gpt2_model_ref.to(device)

gen_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": gpt2_tokenizer.eos_token_id
}

# RL Trainer
ppo_model_trainer = PPOModel(gpt2_model, gpt2_model_ref, gpt2_tokenizer, **config)
total_ppo_epochs = int(np.ceil(config["steps"] / config['batch_size']))

for epoch in tqdm(range(total_ppo_epochs)):
    logs, timing = dict(), dict()
    t0 = time.time()

    batch = {
        'tokens': [],
        'query': []
    }
    for _ in range(config['batch_size']):
        random_prompt = random.choice(prompts)  # 随机选择一个prompt
        tokens = gpt2_tokenizer.encode(random_prompt)
        batch['tokens'].append(tokens)
        batch['query'].append(random_prompt)
    query_tensors = [torch.tensor(t).long().to(device) for t in batch["tokens"]]

    t = time.time()
    response_tensors = []
    for i in range(config['batch_size']):
        gen_len = config['gen_len']
        response = gpt2_model.generate(query_tensors[i].unsqueeze(dim=0),  # generate()用于直接生成token_id
                                       max_new_tokens=gen_len, **gen_kwargs)
        response_tensors.append(response.squeeze()[-gen_len:])
    batch['response'] = [gpt2_tokenizer.decode(r.squeeze()) for r in response_tensors]
    timing['time/get_response'] = time.time() - t

    t = time.time()
    texts = [q + r for q, r in zip(batch['query'], batch['response'])]  # 计算正向/负向情感得分
    pipe_outputs = sentiment_pipe(texts)
    rewards = []
    for output in pipe_outputs:
        if output['label'] == 'positive (stars 4 and 5)':
            rewards.append(output['score'])
        elif output['label'] == 'negative (stars 1, 2 and 3)':
            rewards.append(1 - output['score'])
        else:
            raise ValueError(f"错误的推理结果{output['label']}.")
    rewards = torch.tensor(rewards).to(device)  # 将正向情感的得分作为生成得分
    timing['time/get_sentiment_preds'] = time.time() - t

    t = time.time()
    stats = ppo_model_trainer.step(query_tensors, response_tensors, rewards)  # PPO Update
    timing['time/optimization'] = time.time() - t

    timing['time/epoch'] = time.time() - t0  # logging
    logs.update(timing)
    logs.update(stats)
    logs['env/reward_mean'] = torch.mean(rewards).cpu().numpy()
    logs['env/reward_std'] = torch.std(rewards).cpu().numpy()
    logs['env/reward_dist'] = rewards.cpu().numpy()
    print(f"epoch {epoch} mean-reward: {logs['env/reward_mean']}")

    print('Random Sample 5 text(s) of model output:')
    for i in range(5):  # 随机打5个生成的结果
        print(f'{i + 1}. {random.choice(texts)}')

    writer.add_scalar('train/reward', logs['env/reward_mean'], epoch)
    for k, v in timing.items():
        writer.add_scalar(k, v, epoch)
    writer.add_scalar('ppo/loss/policy', stats['ppo/loss/policy'], epoch)
    writer.add_scalar('ppo/loss/value', stats['ppo/loss/value'], epoch)
    writer.add_scalar('ppo/policy/entropy', stats['ppo/policy/entropy'], epoch)
    writer.add_scalar('ppo/policy/policykl', stats['ppo/policy/policykl'], epoch)
    writer.record()

    if epoch % config['save_freq'] == 0:
        if not os.path.exists(config['save_dir']):
            os.makedirs(config['save_dir'])
        cur_save_path = os.path.join(
            config['save_dir'], f'model_{epoch}_{round(float(logs["env/reward_mean"]), 2)}'
        )
        ppo_model_trainer.model.save_pretrained(cur_save_path)
        ppo_model_trainer.tokenizer.save_pretrained(cur_save_path)