# coding=UTF-8
# 模型训练的一些工具方法

'''
@File: modul_train_util
@Author: WeiWei
@Time: 2023/3/1
@Email: weiwei_519@outlook.com
@Software: PyCharm
'''

from torch.optim.lr_scheduler import LambdaLR


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
   Warmup预热学习率：先从一个较小的学习率线性增加至原来设置的学习率，再进行学习率的线性衰减

    当 current_step < num_warmup_steps时，
    new_lr =current_step/num_warmup_steps * base_lr
    当current_step >= num_warmup_steps时，
    new_lr =(num_training_steps - current_step) / (num_training_steps -num_warmup_steps) * base_lr

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        # 自定义函数
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)
