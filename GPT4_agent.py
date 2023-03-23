# coding=utf-8
"""
# Project Name: MyChatGPT
# File Name: GPT4_agent
# Author: NSNP577
# Creation at 2023/3/23 10:02
# IDE: PyCharm
# Describe: 
"""

from steamship import Steamship

# initial client for gpt-4 from steamship
'''
If you are in Replit:
1) Get an API key at https://steamship.com/account/api
2) Set the STEAMSHIP_API_KEY Replit Secret
3) Close and reopen this shell so that secrets refresh
'''
client = Steamship(workspace="gpt-4", api_key='E7A97033-8208-452F-945D-782EA5E011AC')
# create text generator from steamship client
generator = client.use_plugin('gpt-4')

# execute generation
task = generator.generate(text="请问你可以介绍一下自己吗？")
task.wait()
print(task.output.blocks[0].text)
