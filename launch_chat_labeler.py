# coding=utf-8
# 启动streamlit环境的chat_labeler.py
"""
# Project Name: MyChatGPT
# File Name: launch_chat_labeler
# Author: NSNP577
# Creation at 2023/3/27 13:58
# IDE: PyCharm
# Describe: 
"""

import sys
from streamlit.web import cli as stcli

if __name__ == '__main__':
    sys.argv = ["streamlit", "run", "answer_rank_labeler.py", "--server.port", "8904"]
    sys.exit(stcli.main())
