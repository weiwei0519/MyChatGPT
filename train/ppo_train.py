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
from reward_model import RewardModel
from model.ppo_model import PPOModel
from utils.training_logger import LoggerWriter
import json

writer = LoggerWriter(log_path='../logs/ppo_train', log_name='PPO-train-Zh')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe_device = 0 if torch.cuda.is_available() else -1
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# GPT基础语言模型
gpt2_model_dir = '../models/CompanyModel0.1-GPT2-Chinese'
# 奖惩模型
reward_model_dir = '../models/CompanyModel0.1-RewardModel-Chinese'
# PPO模型输出路径
ppo_saved_dir = '../models/CompanyModel0.1-PPO-Chinese'

# model config
config = {
    "model_name": gpt2_model_dir,
    "steps": 5000,
    "batch_size": 8,
    "forward_batch_size": 8,
    "ppo_epochs": 10,
    "lr": 1.41e-5,
    "init_kl_coef": 0.2,
    "target": 6,
    "horizon": 10000,
    "gamma": 1,
    "lam": 0.95,
    "cliprange": .2,
    "cliprange_value": .2,
    "vf_coef": .1,
    "gen_len": 256,
    "sample_count": 4,
    "save_freq": 5,
    'save_dir': ppo_saved_dir
}

# prompt池
prompts = [
    '刚收到货，感觉',
    '这部电影很',
    '说实话，真的很',
    '这次购物总的来说体验很'
]

# 加载奖惩模型
# reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_dir)
reward_model = torch.load(os.path.join(reward_model_dir, 'model.pt'))
reward_model.to(device)
# sentiment_pipe = pipeline('sentiment-analysis', model=reward_model, tokenizer=reward_tokenizer, device=pipe_device)

# 加载GPT2文本生成模型
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

# 加载Prompt + Ranked Answer数据集，基于人工排序所动态生成的:prompt-ranked answer。
# 文件结构：[['Q1-A(1,1)','Q1-A(1,2)','Q1-A(1,3)','Q1-A(1,4)','Q1-A(1,5)'],
#          ['Q2-A(2,1)','Q2-A(2,2)','Q2-A(2,3)','Q2-A(2,4)','Q2-A(2,5)'],
#          ...
#         ]
reward_pairs_file = '../datasets/reward_model_dataset/reward_prompt_answer_pairs.json'
content = open(reward_pairs_file, 'r', encoding='utf8').read()
reward_pairs = json.loads(content)
del content

# 开始基于reward_pairs进行PPO训练
for epoch in tqdm(range(total_ppo_epochs)):
    logs, timing = dict(), dict()
    t0 = time.time()

    batch = {
        'prompt': [],
        'prompt_encoding': [],
        'answer': [],
        'answer_encoding': [],
    }
    for _ in range(int(config['batch_size'] / config['sample_count'])):
        random_no = random.choice(list(reward_pairs.keys()))  # 随机选择一个prompt
        prompt = reward_pairs[random_no]["prompt"]
        prompt_encoding = gpt2_tokenizer(text=[prompt],
                                         truncation=True,
                                         pad_to_max_length=True,
                                         padding='max_length',
                                         max_length=config['gen_len'],
                                         return_tensors="pt"
                                         )
        ranked_answers = reward_pairs[random_no]["ranked_answers"]
        for answer in ranked_answers:
            batch['prompt'].append(prompt)
            batch['prompt_encoding'].append(prompt_encoding)
            batch['answer'].append(answer)
            answer_encoding = gpt2_tokenizer(text=[answer],
                                             truncation=True,
                                             pad_to_max_length=True,
                                             padding='max_length',
                                             max_length=config['gen_len'],
                                             return_tensors="pt"
                                             )
            batch['answer_encoding'].append(answer_encoding)
    prompt_tensors = [torch.tensor(t['input_ids']).long().to(device) for t in batch["prompt_encoding"]]
    answer_tensors = [torch.tensor(t['input_ids']).long().to(device) for t in batch["answer_encoding"]]
    del reward_pairs
    timing['time/encode'] = time.time() - t0

    # t = time.time()
    # response_tensors = []
    # for i in range(config['batch_size']):
    #     gen_len = config['gen_len']
    #     response = gpt2_model.generate(prompt_tensors[i].unsqueeze(dim=0),  # generate()用于直接生成token_id
    #                                    max_new_tokens=gen_len, **gen_kwargs)
    #     response_tensors.append(response.squeeze()[-gen_len:])
    # batch['response'] = [gpt2_tokenizer.decode(r.squeeze()) for r in response_tensors]
    # timing['time/get_response'] = time.time() - t

    t1 = time.time()
    # 生成输入输出完整pairs
    rewards = []
    for answer_encoding in tqdm(batch['answer_encoding']):
        reward = reward_model(
            input_ids=answer_encoding['input_ids'].to(device),
            token_type_ids=answer_encoding['token_type_ids'].to(device),
            attention_mask=answer_encoding['attention_mask'].to(device),
            pos_ids=torch.tensor([[i for i in range(answer_encoding['input_ids'].shape[1])]]).to(device)
        )
        rewards.append(reward)
        torch.cuda.empty_cache()
    rewards_tensor = torch.tensor(rewards).float().to(device)
    timing['time/reward'] = time.time() - t1
    # for output in pipe_outputs:
    #     if output['label'] == 'positive (stars 4 and 5)':
    #         rewards.append(output['score'])
    #     elif output['label'] == 'negative (stars 1, 2 and 3)':
    #         rewards.append(1 - output['score'])
    #     else:
    #         raise ValueError(f"错误的推理结果{output['label']}.")
    # rewards = torch.tensor(rewards).to(device)  # 将正向情感的得分作为生成得分
    # timing['time/get_sentiment_preds'] = time.time() - t

    t2 = time.time()
    stats = ppo_model_trainer.step(prompt_tensors, answer_tensors, rewards_tensor)  # PPO Update
    timing['time/optimization'] = time.time() - t2

    timing['time/epoch'] = time.time() - t0  # logging
    logs.update(timing)
    logs.update(stats)
    logs['env/reward_mean'] = torch.mean(rewards).cpu().numpy()
    logs['env/reward_std'] = torch.std(rewards).cpu().numpy()
    logs['env/reward_dist'] = rewards.cpu().numpy()
    print(f"epoch {epoch} mean-reward: {logs['env/reward_mean']}")

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
