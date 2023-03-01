# coding=UTF-8
# 基于ColossalAI的ChatGPT训练

'''
@File: ChatGPT_on_ColossalAI
@Author: WeiWei
@Time: 2023/2/19
@Email: weiwei_519@outlook.com
@Software: PyCharm
'''

from chatgpt.nn import GPTActor, GPTCritic, RewardModel
from chatgpt.trainer import PPOTrainer
from chatgpt.trainer.strategies import ColossalAIStrategy
from copy import deepcopy

strategy = ColossalAIStrategy(stage=3, placement_policy='cuda')
with strategy.model_init_context():
    actor = GPTActor().cuda()
    critic = GPTCritic().cuda()
initial_model = deepcopy(actor).cuda()
reward_model = RewardModel(deepcopy(critic.model)).cuda()
trainer = PPOTrainer(strategy, actor, critic, reward_model, initial_model, ...)
trainer.fit(prompts)