# coding=UTF-8
# Rank List 标注平台，用于标注 Reward Model 的训练数据，通过streamlit搭建。

'''
@File: answer_rank_labeler
@Author: WeiWei
@Time: 2023/3/5
@Email: weiwei_519@outlook.com
@Software: PyCharm
'''

import os
import random

import numpy as np
import pandas as pd
import streamlit as st
import torch
import json
from transformers import AutoTokenizer, GPT2LMHeadModel, TextGenerationPipeline

st.set_page_config(
    page_title="Rank List Labeler",
    page_icon='📌',
    layout="wide"
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODEL_CONFIG = {
    'model_name': './models/chatgpt-aia-chinese',  # backbone
    'dataset_file': './datasets/company_datasets/human_rank_pairs.json',  # 标注数据集的存放文件
    'rank_list_len': 4,  # 排序列表的长度
    'max_gen_seq_len': 300,  # 生成答案最大长度
    'random_prompts': [  # 随机prompt池
        '盛世经典尊享版终身寿险的职业限制',
        '友童乐齿医疗保险，在保险合同有效期内',
        '传世金生年金保险(分红型)的年金处理方式',
        '友童乐齿医疗保险是补偿型医疗保险'
    ]
}

######################## 页面配置初始化 ###########################
RANK_COLOR = [
    'red',
    'green',
    'blue',
    'orange',
    'violet'
]

######################## 会话缓存初始化 ###########################
if 'model_config' not in st.session_state:
    st.session_state['model_config'] = MODEL_CONFIG

if 'model' not in st.session_state:
    model_name = st.session_state['model_config']['model_name']
    st.session_state['model'] = GPT2LMHeadModel.from_pretrained(model_name)

if 'tokenizer' not in st.session_state:
    model_name = st.session_state['model_config']['model_name']
    st.session_state['tokenizer'] = AutoTokenizer.from_pretrained(model_name)

if 'generator' not in st.session_state:
    st.session_state['generator'] = TextGenerationPipeline(
        st.session_state['model'],
        st.session_state['tokenizer'],
        device=device
    )

if 'current_results' not in st.session_state:
    st.session_state['current_results'] = [''] * MODEL_CONFIG['rank_list_len']

if 'current_prompt' not in st.session_state:
    st.session_state['current_prompt'] = '今天早晨我去了'


######################### 函数定义区 ##############################
def generate_text():
    """
    模型生成文字。
    """
    current_results = []
    for _ in range(MODEL_CONFIG['rank_list_len']):
        res = st.session_state['generator'](
            st.session_state['current_prompt'],
            max_length=MODEL_CONFIG['max_gen_seq_len'],
            do_sample=True
        )
        current_results.extend([e['generated_text'] for e in res])
    st.session_state['current_results'] = current_results


######################### 页面定义区（侧边栏） ########################
st.sidebar.title('📌 Rank List 标注平台')
st.sidebar.markdown('''
    ```python
    用于生成模型生成 Rank List 的标注。
    ```
''')
st.sidebar.markdown('标注思路参考自 [InstructGPT](https://arxiv.org/pdf/2203.02155.pdf) 。')
st.sidebar.markdown('RLHF 更多介绍：[想训练ChatGPT？得...](https://zhuanlan.zhihu.com/p/595579042)')
st.sidebar.header('⚙️ Model Config')
st.sidebar.write('当前标注配置（可在源码中修改）：')
st.sidebar.write(st.session_state['model_config'])

label_tab, dataset_tab = st.tabs(['Label', 'Dataset'])

######################### 页面定义区（标注页面） ########################
with label_tab:
    with st.expander('🔍 Setting Prompts', expanded=True):
        random_button = st.button('随机 prompt',
                                  help='从prompt池中随机选择一个prompt，可通过修改源码中 MODEL_CONFIG["random_prompts"] 参数来自定义prompt池。')
        if random_button:
            prompt_text = random.choice(MODEL_CONFIG['random_prompts'])
        else:
            prompt_text = st.session_state['current_prompt']

        query_txt = st.text_input('prompt: ', prompt_text)
        if query_txt != st.session_state['current_prompt']:
            st.session_state['current_prompt'] = query_txt
            generate_text()

    with st.expander('💡 Generate Results', expanded=True):
        if st.session_state['current_results'][0] == '':
            generate_text()

        columns = st.columns([1] * MODEL_CONFIG['rank_list_len'])
        rank_results = [-1] * MODEL_CONFIG['rank_list_len']
        rank_choices = [-1] + [i + 1 for i in range(MODEL_CONFIG['rank_list_len'])]
        for i, c in enumerate(columns):
            with c:
                choice = st.selectbox(f'句子{i + 1}排名', rank_choices, help='为当前句子选择排名，排名越小，得分越高。（-1代表当前句子暂未设置排名）')
                if choice != -1 and choice in rank_results:
                    st.info(f'当前排名[{choice}]已经被句子[{rank_results.index(choice) + 1}]占用，请先将占用排名的句子置为-1再为当前句子分配该排名。')
                else:
                    rank_results[i] = choice
                color = RANK_COLOR[i] if i < len(RANK_COLOR) else 'white'
                # st.write(color)
                st.markdown(f":{color}[{st.session_state['current_results'][i]}]")

    with st.expander('🥇 Rank Results', expanded=True):
        columns = st.columns([1] * MODEL_CONFIG['rank_list_len'])
        for i, c in enumerate(columns):
            with c:
                st.write(f'Rank {i + 1}：')
                if i + 1 in rank_results:
                    color = RANK_COLOR[rank_results.index(i + 1)] if rank_results.index(i + 1) < len(
                        RANK_COLOR) else 'white'
                    st.markdown(f":{color}[{st.session_state['current_results'][rank_results.index(i + 1)]}]")

    save_button = st.button('存储当前排序')
    if save_button:
        dataset_file_name = MODEL_CONFIG['dataset_file'].split('/')[-1]
        dataset_file_path = MODEL_CONFIG['dataset_file'].replace(dataset_file_name, '')
        if not os.path.exists(dataset_file_path):
            os.makedirs(dataset_file_path)

        if -1 in rank_results:
            st.error('请完成排序后再存储！', icon='🚨')
            st.stop()

        with open(MODEL_CONFIG['dataset_file'], 'a', encoding='utf8') as f:
            rank_texts = []
            for i in range(len(rank_results)):
                rank_texts.append(st.session_state['current_results'][rank_results.index(i + 1)])
            line = '\t'.join(rank_texts)
            f.write(f'{line}\n')

        st.success('保存成功，请更换prompt生成新的答案~', icon="✅")

######################### 页面定义区（数据集页面） #######################
with dataset_tab:
    file = open(MODEL_CONFIG['dataset_file'], 'w+', encoding='utf8')
    content = file.read()
    if len(content) != 0 and content != '':
        rank_pairs = json.loads(content)
    else:
        rank_pairs = {}

    # dumps()：将dict数据转化成json数据；   dump()：将dict数据转化成json数据后写入json文件
    # rank_pairs = json.dumps(rank_pairs)
    json.dump(rank_pairs, file)