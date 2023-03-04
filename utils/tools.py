# coding=utf-8
# 工具类
"""
# Project Name: MyChatGPT
# File Name: tools
# Author: NSNP577
# Creation at 2023/3/3 17:07
# IDE: PyCharm
# Describe: 
"""
import os
import traceback

import numpy as np
from rich import print


def get_project_path():
    return os.getcwd().replace("\\", "/")
