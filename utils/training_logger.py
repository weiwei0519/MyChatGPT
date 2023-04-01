# coding=utf-8
# 训练过程中的记录器，类似于SummaryWriter功能，只是SummaryWriter需要依赖于tensorboard和浏览器做可视化，
# 该工具依赖matplotlib采用静态本地图片存储的形式，便于服务器快速查看训练结果。
"""
# Project Name: MyChatGPT
# File Name: training_logger
# Author: NSNP577
# Creation at 2023/3/2 11:42
# IDE: PyCharm
# Describe:
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import logging


class LoggerWriter(object):

    def __init__(self, log_path: str, log_name: str, params=[], extention='.png', max_columns=2,
                 log_title=None, figsize=None):
        """
        初始化函数，创建日志类。

        Args:
            log_path (str): 日志存放文件夹
            log_name (str): 日志文件名
            parmas (list): 要记录的参数名字列表，e.g. -> ["loss", "reward", ...]
            extension (str): 图片存储格式
            max_columns (int): 一行中排列几张图，默认为一行2张（2个变量）的图。
        """
        self.log_path = log_path
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.log_name = log_name
        self.extention = extention
        self.max_param_index = -1
        self.max_columns_threshold = max_columns
        self.figsize = figsize
        self.params_dict = self.create_params_dict(params)
        self.log_title = log_title
        self.init_plt()
        self.update_ax_list()

    def init_plt(self) -> None:
        plt.style.use('seaborn-darkgrid')

    def create_params_dict(self, params: list) -> dict:
        """
        根据传入需要记录的变量名列表，创建监控变量字典。

        Args:
            params (list): 监控变量名列表

        Returns:
            dict: 监控变量名字典 -> {
                'loss': {'values': [0.44, 0.32, ...], 'epochs': [10, 20, ...], 'index': 0},
                'reward': {'values': [10.2, 13.2, ...], 'epochs': [10, 20, ...], 'index': 1},
                ...
            }
        """
        params_dict = {}
        for i, param in enumerate(params):
            params_dict[param] = {'values': [], 'epochs': [], 'index': i}
            self.max_param_index = i
        return params_dict

    def update_ax_list(self) -> None:
        """
        根据当前的监控变量字典，为每一个变量分配一个图区。
        """
        # * 重新计算每一个变量对应的图幅索引
        params_num = self.max_param_index + 1
        if params_num <= 0:
            return

        self.max_columns = params_num if params_num < self.max_columns_threshold else self.max_columns_threshold
        max_rows = (params_num - 1) // self.max_columns + 1  # * 所有变量最多几行
        figsize = self.figsize if self.figsize else (self.max_columns * 6, max_rows * 3)  # 根据图个数计算整个图的figsize
        self.fig, self.axes = plt.subplots(max_rows, self.max_columns, figsize=figsize)

        # * 如果只有一行但又不止一个图，需要手动reshape成(1, n)的形式
        if params_num > 1 and len(self.axes.shape) == 1:
            self.axes = np.expand_dims(self.axes, axis=0)

        # * 重新设置log标题
        log_title = self.log_title if self.log_title else '[Training Log] {}'.format(
            self.log_name)
        self.fig.suptitle(log_title, fontsize=15)

    def add_scalar(self, param: str, value: float, epoch: int) -> None:
        """
        添加一条新的变量值记录。

        Args:
            param (str): 变量名，e.g. -> 'loss'
            value (float): 此时的值。
            epoch (int): 此时的epoch数。
        """
        # * 如果该参数是第一次加入，则将该参数加入到监控变量字典中
        if param not in self.params_dict:
            self.max_param_index += 1
            self.params_dict[param] = {'values': [],
                                       'epochs': [], 'index': self.max_param_index}
            self.update_ax_list()

        self.params_dict[param]['values'].append(value)
        self.params_dict[param]['epochs'].append(epoch)

    def record(self, dpi=200) -> None:
        """
        调用该接口，对该类中目前所有监控的变量状态进行一次记录，将结果保存到本地文件中。
        """
        for param, param_elements in self.params_dict.items():
            param_index = param_elements["index"]
            param_row, param_column = param_index // self.max_columns, param_index % self.max_columns
            ax = self.axes[param_row, param_column] if self.max_param_index > 0 else self.axes
            # ax.set_title(param)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(param)
            ax.plot(self.params_dict[param]['epochs'],
                    self.params_dict[param]['values'],
                    color='darkorange')

        plt.savefig(os.path.join(self.log_path,
                                 self.log_name + self.extention), dpi=dpi)


def get_logger(log_path, log_name):
    '''
    配置log
    :param log_path: 输出log路径
    :param logging_name: 记录中name，可随意
    :return:
    '''
    '''
    logger是日志对象，handler是流处理器，console是控制台输出（没有console也可以，将不会在控制台输出，会在日志文件中输出）
    '''
    # 获取logger对象,取名
    logger = logging.getLogger(log_name)
    # 输出DEBUG及以上级别的信息，针对所有输出的第一层过滤
    logger.setLevel(level=logging.DEBUG)
    # 获取文件日志句柄并设置日志级别，第二层过滤
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    # 生成并设置文件日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # console相当于控制台输出，handler文件输出。获取流句柄并设置日志级别，第二层过滤
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # 为logger对象添加句柄
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


if __name__ == '__main__':
    n_epochs = 10
    log_path, log_name = './logs', 'xxx_test'
    displayer = LoggerWriter(log_path=log_path, log_name=log_name)
    for i in range(n_epochs):
        loss, reward = 100 - random.random() * i, random.random() * i
        displayer.add_scalar('loss', loss, i)
        displayer.add_scalar('reward', reward, i)
        displayer.add_scalar('random', reward, i)
        displayer.record()
        print("Log has been saved at: {}".format(os.path.join(log_path, log_name)))
        time.sleep(3)
