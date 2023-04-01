# coding=utf-8
# 面向IT的开发辅助工具。基于chatglm
"""
# Project Name: MyChatGPT
# File Name: chatglm_web
# Author: NSNP577
# Creation at 2023/3/29 19:45
# IDE: PyCharm
# Describe: 
"""

import os
import streamlit as st
import torch
import json
from PIL import Image
import logging
import datetime
from transformers import AutoTokenizer, GPT2LMHeadModel, TextGenerationPipeline, AutoConfig, AutoModel
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from model.T5_model import infer_answer
from steamship import Steamship
# from streamlit_autorefresh import st_autorefresh
import urllib3

urllib3.disable_warnings()

today = datetime.datetime.now().strftime("%Y-%m-%d")
logging.basicConfig(filename=f'./logs/labeler/labeler_{today}.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filemode='a'
                    )

st.set_page_config(
    page_title="AIA-GPT 管理平台",
    page_icon="🦈",
    layout="wide"
)

GPU = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CPU = torch.device("cpu")

MODEL_CONFIG = {
    'model_AIA': './models/chatgpt-aia-chinese/ttt-aia-chinese',  # backbone
    'model_glm': './models/THUDM/chatglm-6b-int4',  # backbone
    'dataset_file': './datasets/company_datasets/human_rank_pairs.json',  # 标注数据集的存放文件
    'rank_list_len': 3,  # 排序列表的长度
    'max_gen_seq_len': 400,  # 生成答案最大长度
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

if 'prompt' not in st.session_state:
    st.session_state['prompt'] = ''

if 'query' not in st.session_state:
    st.session_state['query'] = ''

if 'answer' not in st.session_state:
    st.session_state['answer'] = ''

if 'answer_to_rank' not in st.session_state:
    answer_to_rank = {}
    st.session_state['answer_to_rank'] = answer_to_rank

if 'rank_choices' not in st.session_state:
    rank_choices = ['-1']
    for i in range(MODEL_CONFIG['rank_list_len']):
        rank_choices.append(str(i + 1))
    st.session_state['rank_choices'] = rank_choices

# 加载GPT4 client
if 'GPT4_client' not in st.session_state:
    # client = Steamship(workspace="gpt-4", api_key='E7A97033-8208-452F-945D-782EA5E011AC', verify=False)
    # # create text generator from steamship client
    # generator = client.use_plugin('gpt-4')
    # st.session_state['GPT4_client'] = generator
    st.session_state['GPT4_client'] = ''

if 'program_prompt' not in st.session_state:
    st.session_state['program_prompt'] = ''

if 'program_answer' not in st.session_state:
    st.session_state['program_answer'] = ''


######################### 函数定义区 ##############################
def generate_from_GPT4(prompt):
    logging.info(f'GPT4 prompt: {prompt}')
    generator = st.session_state['GPT4_client']
    task = generator.generate(text=prompt)
    task.wait()
    response = task.output.blocks[0].text
    # print(f'GPT4 answer is: {response}')
    st.session_state['program_answer'] = response


def generate_from_ChatGLM(prompt):
    logging.info('加载ChatGLM预训练模型')
    tokenizer_glm = AutoTokenizer.from_pretrained(MODEL_CONFIG['model_glm'], trust_remote_code=True)
    config_glm = AutoConfig.from_pretrained(MODEL_CONFIG['model_glm'], trust_remote_code=True)
    # model_glm = AutoModel.from_pretrained(MODEL_CONFIG['model_glm'],
    #                                       trust_remote_code=True,
    #                                       ignore_mismatched_sizes=True).float()  # CPU
    model_glm = AutoModel.from_pretrained(MODEL_CONFIG['model_glm'],
                                          trust_remote_code=True,
                                          ignore_mismatched_sizes=True).half().cuda()  # GPU
    logging.info('ChatGLM预训练模型加载成功')
    logging.info(f'ChatGLM prompt: {prompt}')
    response, _ = model_glm.chat(tokenizer_glm, prompt, history=[])
    st.session_state['program_answer'] = response
    del model_glm, config_glm, tokenizer_glm
    torch.cuda.empty_cache()


######################### 页面定义区（侧边栏） ########################
with st.sidebar:
    st.image(Image.open('./images/AIA_logo.png'))
    st.sidebar.title('📌 AIA-GPT 管理平台')
    st.sidebar.markdown('''
    ```python
    提供基于GPT模型的生成式内容提问:

    Tab 1:
        Prompt Tab：问答模式。
            用户输入提示问题，系统输出唯一答案。

    Tab 2:
        Label Tab：标注模式。
            用户输入提示问题，系统输出多个答案。
            用户根据答案的准确性进行排序。

    Tab 3:
        Program：代码辅助工具
            基于需求生成代码。
            基于代码，提供解读。
            代码bug发现与修复建议。
            代码单元测试建议。
            为既有代码添加注释。

    ```
    ''')

prompt_tab, label_tab, program_tab = st.tabs(['Program'])
answer_to_rank = []
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: left;} </style>',
         unsafe_allow_html=True)

# st.write('<style>div.st-bf{flex-direction:row;} div.st-ag{font-weight:bold;padding-left:2px;}</style>',
#          unsafe_allow_html=True)

######################### 页面定义区（代码辅助工具页面） ########################
with program_tab:
    st.subheader(':blue[提供面向IT开发人员的代码辅助工具]')
    with st.expander('', expanded=True):  # expanded 是展开还是收缩状态
        func_options = ('请选择：', '生成代码', '代码解读', '修复bug', '单元测试', '添加注释')
        # gene_tab, read_tab, bug_tab, test_tab, comment_tab = st.tabs(func_options)
        func_radio = st.radio(label='请选择您的需求：',
                              options=func_options,
                              index=0,
                              key='func_options',
                              )
        lang_options = ('Java', 'SQL', 'Vue.js', 'Python', 'C++', 'Cobol', 'Swift')
        lang_radio = st.radio(label='请选择您代码的语言：',
                              options=lang_options,
                              index=0,
                              key='lang_options_gene',
                              )
        prompt = ""
        if func_radio == '生成代码':
            prompt = "请帮我生成一段" + lang_radio + "代码，需求是：\n"
        elif func_radio == '代码解读':
            prompt = "请帮我总结如下" + lang_radio + "代码的用途，代码是：\n"
        elif func_radio == '修复bug':
            prompt = "请帮我发现并修复如下" + lang_radio + "代码中存在的问题，代码是：\n"
        elif func_radio == '单元测试':
            prompt = "请帮我生成如下" + lang_radio + "代码的单元测试，代码是：\n"
        elif func_radio == '添加注释':
            prompt = "请帮我为如下" + lang_radio + "代码添加注释，代码是：\n"
        program_request = st.text_area(label='🔍 选择哪种程序语言后，请再输入您的详细需求：',
                                       placeholder='此处输入您的详细需求',
                                       value=prompt,
                                       height=200,
                                       key='program_prompt')
        ask = st.button('提问')
        if ask and program_request != '':
            last_program_prompt = st.session_state['program_prompt']
            # st.session_state['program_prompt'] = program_request
            # generate_from_GPT4(program_request)
            st.session_state.disabled = True  # 防止重复提交，节省内存
            generate_from_ChatGLM(program_request)

        if 'program_answer' in st.session_state and st.session_state['program_answer'] != '':
            answer = st.session_state['program_answer']
            st.text("💡 GPT的回答如下：")
            color = RANK_COLOR[0]
            st.markdown(f":{color}[{answer}]")

        st.session_state.disabled = False