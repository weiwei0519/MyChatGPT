# coding=utf-8
# GPT模型采样输出后，进行奖励模型的训练

"""
# Project Name: MyChatGPT
# File Name: reward_model
# Author: NSNP577
# Creation at 2023/3/2 11:42
# IDE: PyCharm
# Describe: 
"""

import torch
from torch import nn
from torch import sigmoid
from rich import print
from transformers import AutoModel, AutoTokenizer
from typing import List

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RewardModel(nn.Module):

    def __init__(self, encoder):
        """
        init func.

        Args:
            encoder (transformers.AutoModel): backbone, 默认使用 ernie 3.0
        """
        super().__init__()
        self.encoder = encoder
        self.reward_layer = nn.Linear(768, 1)  # reward layer 用于映射到 1 维 reward

    def forward(
            self,
            input_ids: torch.tensor,
            token_type_ids: torch.tensor,
            attention_mask=None,
            pos_ids=None,
    ) -> torch.tensor:
        """
        forward 函数，返回每句话的得分值。

        Args:
            input_ids (torch.tensor): (batch, seq_len)
            token_type_ids (torch.tensor): (batch, seq_len)
            attention_mask (torch.tensor): (batch, seq_len)
            pos_ids (torch.tensor): (batch, seq_len)

        Returns:
            reward: (batch, 1)
        """
        encoder_output = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=pos_ids,
            attention_mask=attention_mask,
        )["pooler_output"]  # (batch, hidden_size)
        encoder_output = torch.squeeze(encoder_output)
        reward = self.reward_layer(encoder_output)  # (batch, 1)
        return reward


def compute_rank_list_loss(rank_rewards_list: List[List[torch.tensor]], device=device) -> torch.Tensor:
    """
    通过给定的有序（从高到低）的ranklist的reward列表，计算rank loss。
    所有排序高的句子的得分减去排序低的句子的得分差的总和，并取负。

    Args:
        rank_rewards_list (torch.tensor): 有序（从高到低）排序句子的reward列表，e.g. ->
                                        [
                                            [torch.tensor([0.3588]), torch.tensor([0.2481]), ...],
                                            [torch.tensor([0.5343]), torch.tensor([0.2442]), ...],
                                            ...
                                        ]
        device (str): 使用设备

    Returns:
        loss (torch.tensor): tensor([0.4891], grad_fn=<DivBackward0>)
    """
    if type(rank_rewards_list) != list:
        raise TypeError(f'@param rank_rewards expected "list", received {type(rank_rewards_list)}.')

    loss, add_count = torch.tensor([0]).to(device), 0
    for rank_rewards in rank_rewards_list:
        for i in range(len(rank_rewards) - 1):  # 遍历所有前项-后项的得分差
            for j in range(i + 1, len(rank_rewards)):
                diff = sigmoid(rank_rewards[i] - rank_rewards[j])  # sigmoid到0~1之间
                loss = loss + diff
                add_count += 1
    loss = loss / add_count
    return -loss


if __name__ == '__main__':
    encoder = AutoModel.from_pretrained('./models/CompanyModel0.1-GPT2-Chinese')
    model = RewardModel(encoder)
    tokenizer = AutoTokenizer.from_pretrained('./models/CompanyModel0.1-GPT2-Chinese')

    batch_texts = [
        ['这是一个测试句子1。', '这是一个测试句子2。', '这是一个测试句子3。', '这是一个测试句子4。'],
        ['这是一个测试句子5。', '这是一个测试句子6。', '这是一个测试句子7。', '这是一个测试句子8。'],
    ]

    rank_rewards = []
    for texts in batch_texts:
        tmp = []
        for text in texts:
            inputs = tokenizer(text, return_tensors='pt')
            r = model(**inputs)
            tmp.append(r[0])
        rank_rewards.append(tmp)
    print('rank_rewards: ', rank_rewards)
    loss = compute_rank_list_loss(rank_rewards)
    print('loss: ', loss.item())
    loss.backward()
