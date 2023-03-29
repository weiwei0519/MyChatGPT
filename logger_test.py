# coding=utf-8
"""
# Project Name: MyChatGPT
# File Name: logger_test
# Author: NSNP577
# Creation at 2023/3/28 13:43
# IDE: PyCharm
# Describe: 
"""

import logging
import datetime

today = datetime.datetime.now().strftime("%Y-%m-%d")
logging.basicConfig(filename=f'./logs/test/test_{today}.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filemode='a'
                    )
logging.info('this is logging message')




