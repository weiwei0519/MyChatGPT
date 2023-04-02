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
import streamlit as st
import torch
import json
from PIL import Image
import logging
import datetime
from transformers import AutoTokenizer, GPT2LMHeadModel, TextGenerationPipeline
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from model.T5_model import infer_answer
from steamship import Steamship
# from streamlit_autorefresh import st_autorefresh
import requests as req
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODEL_CONFIG = {
    'model_name': './models/chatgpt-aia-chinese/ttt-aia-chinese',  # backbone
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

# 加载预训练模型：
# tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG['model_name'])
# model = GPT2LMHeadModel.from_pretrained(MODEL_CONFIG['model_name'])
# TextGenerationPipeline(model, tokenizer, device=device)
# model.to(device)

# 加载Company预训练模型
# if 'model' not in st.session_state:
#     tokenizer = T5Tokenizer.from_pretrained(MODEL_CONFIG['model_name'])
#     config = T5Config.from_pretrained(MODEL_CONFIG['model_name'])
#     model = T5ForConditionalGeneration.from_pretrained(MODEL_CONFIG['model_name'])
#     model.to(device)
#     st.session_state['tokenizer'] = tokenizer
#     st.session_state['config'] = config
#     st.session_state['model'] = model

######################## 会话缓存初始化 ###########################

if 'aiagpt_prompt' not in st.session_state:
    st.session_state['aiagpt_prompt'] = ''

if 'aiagpt_query' not in st.session_state:
    st.session_state['aiagpt_query'] = ''

if 'aiagpt_answer' not in st.session_state:
    st.session_state['aiagpt_answer'] = ''

if 'chatgpt_prompt' not in st.session_state:
    st.session_state['chatgpt_prompt'] = ''

if 'chatgpt_query' not in st.session_state:
    st.session_state['chatgpt_query'] = ''

if 'chatgpt_answer' not in st.session_state:
    st.session_state['chatgpt_answer'] = ''

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

if 'chatgpt_program_prompt' not in st.session_state:
    st.session_state['chatgpt_program_prompt'] = ''

if 'chatgpt_program_answer' not in st.session_state:
    st.session_state['chatgpt_program_answer'] = ''


######################### 函数定义区 ##############################
def generate_text(prompt, do_sample):
    """
    模型生成文字。
    """
    logging.info(f'AIA-GPT prompt: {prompt}')
    logging.info('加载AIAGPT model')
    tokenizer = T5Tokenizer.from_pretrained(MODEL_CONFIG['model_name'])
    model = T5ForConditionalGeneration.from_pretrained(MODEL_CONFIG['model_name'])
    model.to(device)
    logging.info('AIAGPT model加载成功')
    # current_results = []
    # for _ in range(MODEL_CONFIG['rank_list_len']):
    #     if st.session_state['current_prompt'] != '':
    #         res = st.session_state['generator'](
    #             st.session_state['current_prompt'],
    #             max_length=MODEL_CONFIG['max_gen_seq_len'],
    #             do_sample=True,
    #             early_stopping=True,  # 保证当遇到EOS词时，结束生成
    #         )
    #         current_results.extend([e['generated_text'] for e in res])
    # st.session_state['current_results'] = current_results
    return_seqs = MODEL_CONFIG['rank_list_len']
    if do_sample:
        # 标注模式
        action = '标注模式'
        return_seqs = MODEL_CONFIG['rank_list_len']
        logging.info(f'action: {action}')
    else:
        # 问答模式
        action = '问答模式'
        return_seqs = 1
        logging.info(f'action: {action}')
    answers = infer_answer(text=prompt,
                           tokenizer=tokenizer,
                           model=model,
                           # do_sample=do_sample,
                           samples=return_seqs,
                           out_length=MODEL_CONFIG['max_gen_seq_len'],
                           )
    answers = [answer.replace('\n', '<br>').replace('\t', '   ') for answer in answers]
    logging.info(f'answer in generator: {len(answers)} x {len(answers[0])}')
    logging.info(f'answer in generator: {answers}')
    if action == '标注模式':
        st.session_state['answer_to_rank'] = {}
        for _, answer in enumerate(answers):
            st.session_state['answer_to_rank'][answer] = '-1'  # -1 为初始rank，等于未排序
    elif action == '问答模式':
        st.session_state['aiagpt_answer'] = answers[0]
    del model, tokenizer
    torch.cuda.empty_cache()


def generate_from_GPT4(prompt):
    logging.info(f'GPT4 prompt: {prompt}')
    # generator = st.session_state['GPT4_client']
    # task = generator.generate(text=prompt)
    # task.wait()
    # response = task.output.blocks[0].text
    # print(f'GPT4 answer is: {response}')
    st.session_state['gpt4_program_answer'] = ''


def generate_from_GPT3_from_JJY(prompt):
    URL = 'https://openapi.jijyun.cn/api/openapi/corp_token'
    req_body = {
        "corp_id": "gLnZxt6M21cPBoV6wIT6xYBNTIhv56Av",
        "secret": "fE6btX6T6GluuN6b5B7MWckV2wZtYv5w",
        "token": prompt
    }

    response = req.post(url=URL, data=req_body)
    st.session_state['chatgpt_answer'] = response


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

AIAGPT_prompt_tab, AIAGPT_label_tab, ChatGPT_prompt_tab, ChatGPT_program_tab = st.tabs(
    ['AIA-GPT-Prompt', 'AIA-GPT-Label', 'ChatGPT-Prompt', 'ChatGPT-Program'])
answer_to_rank = []
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: left;} </style>',
         unsafe_allow_html=True)

# st.write('<style>div.st-bf{flex-direction:row;} div.st-ag{font-weight:bold;padding-left:2px;}</style>',
#          unsafe_allow_html=True)

######################### 页面定义区（问答页面） ########################
with AIAGPT_prompt_tab:
    st.subheader(':blue[AIA-GPT对话问答]')
    with st.expander('', expanded=True):  # expanded 是展开还是收缩状态
        qa_query_txt = st.text_area('🔍 请输入您的提问：', placeholder='此处输入您的提问', key='qa_query')
        if qa_query_txt != '' and st.session_state['aiagpt_prompt'] != qa_query_txt:
            st.session_state['aiagpt_prompt'] = qa_query_txt
            generate_text(st.session_state['aiagpt_prompt'], do_sample=False)

    with st.expander('💡 GPT的回答如下：', expanded=True):
        if 'aiagpt_answer' in st.session_state:
            answer = st.session_state['aiagpt_answer']
            if answer != '':
                color = RANK_COLOR[0]
                st.markdown(f":{color}[{answer}]", unsafe_allow_html=True)

######################### 页面定义区（标注页面） ########################
with AIAGPT_label_tab:
    st.subheader(':blue[AIA-GPT对话问答]')
    with st.expander('', expanded=True):  # expanded 是展开还是收缩状态
        label_query_txt = st.text_area('🔍 请输入您的提问：', placeholder='此处输入您的提问', key='label_query')
        if label_query_txt != '' and st.session_state['aiagpt_query'] != label_query_txt:
            st.session_state['aiagpt_query'] = label_query_txt
            generate_text(st.session_state['aiagpt_query'], do_sample=True)

    with st.expander('💡 GPT的回答如下：', expanded=True):
        answer_to_rank = st.session_state['answer_to_rank']
        if len(answer_to_rank) > 0:
            rank_choices = st.session_state['rank_choices']
            i = 0
            for answer, rank in answer_to_rank.items():
                if rank != '0':
                    rank_col, answer_col = st.columns([1, 10])
                    i += 1
                    curr_choice = int(rank) if rank != '-1' else 0
                    with rank_col:
                        st.text("🥇 Choose Rank")
                        choice = st.selectbox(f'句子{i}排名', rank_choices,
                                              help='为当前句子选择排名，排名越小，得分越高。（-1代表当前句子暂未设置排名）')
                        if choice != curr_choice:
                            st.session_state['answer_to_rank'][answer] = choice
                            curr_choice = choice
                    with answer_col:
                        st.text("💡 Generated Answers")
                        if curr_choice != '-1':
                            color = RANK_COLOR[int(curr_choice) - 1]
                            st.markdown(f":{color}[{answer}]")
                        else:
                            color = 'white'
                            st.markdown(f":{color}[{answer}]", unsafe_allow_html=True)
                # st.session_state['answer_to_rank'] = answer_to_rank

            # 手工写答案
            rank_col, answer_col = st.columns([1, 10])
            rank = MODEL_CONFIG['rank_list_len'] + 1
            with rank_col:
                st.text("🥇 Choose Rank")
                st.text('0')
            with answer_col:
                input_answer = st.text_area(label='💡 请输入您的答案：', placeholder='这里手工输入答案')
                add_button = st.button('＋添加手工答案')
                if input_answer != '' and add_button:
                    st.session_state['answer_to_rank'][input_answer] = '0'
                    st.success('成功添加手工答案，请点击按钮存储当前排序', icon="✅")

        else:
            rank_col, answer_col = st.columns([1, 10])
            with rank_col:
                st.text("🥇 Choose Rank")
            with answer_col:
                st.text("💡 Generate Answers")

    answer_to_rank_new = st.session_state['answer_to_rank']

    save_button = st.button('存储当前排序')
    if save_button:
        answer_to_rank = st.session_state['answer_to_rank']
        rank_list = []
        for ans, rank in answer_to_rank.items():
            if rank == '-1':
                st.error('请完成排序后再存储！', icon='🚨')  # 所有标注都已排序，才能保存
                st.stop()
            if rank in rank_list:
                st.error('序号选择重复！', icon='🚨')  # 序号选择重复
                st.stop()
            else:
                rank_list.append(rank)

        file = open(MODEL_CONFIG['dataset_file'], 'r+', encoding='utf8')
        content = file.read()
        # logging.info(f'orignal file content: {content}')
        file.seek(0)
        file.truncate()
        if len(content) != 0 and content != '':
            rank_pairs = json.loads(content)
        else:
            rank_pairs = {}
        length = len(rank_pairs)
        logging.info(f'orignal json length: {length}')
        rank_pairs[length] = {}
        rank_pairs[length]['prompt'] = st.session_state['prompt'].strip()
        ranking = {}
        for ans, rank in answer_to_rank.items():
            ranking[rank] = ans
        ranked_answers = [ranking[key] for key in sorted(ranking.keys())]
        rank_pairs[length]['ranked_answers'] = ranked_answers
        # dumps()：将dict数据转化成json数据；   dump()：将dict数据转化成json数据后写入json文件
        content = json.dumps(rank_pairs, ensure_ascii=False, indent=2)
        logging.info(f'file save content length {len(content)}')
        # json.dump(content, file, ensure_ascii=False, indent=2)
        file.write(content)
        file.flush()
        st.success('保存成功，请更换prompt生成新的答案~', icon="✅")
        st.session_state.clear()
        # label_tab.empty()
        # st._rerun()
        # st_autorefresh(interval=2000, limit=100, key="fizzbuzzcounter")

######################### 页面定义区（ChatGPT API页面） ########################
with ChatGPT_prompt_tab:
    st.subheader(':blue[ChatGPT对话问答]')
    with st.expander('', expanded=True):  # expanded 是展开还是收缩状态
        cg_qa_query_txt = st.text_area('🔍 请输入您的提问：', placeholder='此处输入您的提问', key='qa_query')
        if cg_qa_query_txt != '' and st.session_state['chatgpt_prompt'] != cg_qa_query_txt:
            st.session_state['chatgpt_prompt'] = cg_qa_query_txt
            generate_from_GPT3_from_JJY(st.session_state['chatgpt_prompt'])

    with st.expander('💡 GPT的回答如下：', expanded=True):
        if 'chatgpt_answer' in st.session_state:
            answer = st.session_state['chatgpt_answer']
            if answer != '':
                color = RANK_COLOR[0]
                st.markdown(f":{color}[{answer}]", unsafe_allow_html=True)

######################### 页面定义区（代码辅助工具页面） ########################
with ChatGPT_program_tab:
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
            last_program_prompt = st.session_state['chatgpt_program_prompt']
            # st.session_state['program_prompt'] = program_request
            generate_from_GPT4(program_request)

        if 'chatgpt_program_answer' in st.session_state and st.session_state['chatgpt_program_answer'] != '':
            answer = st.session_state['chatgpt_program_answer']
            st.text("💡 GPT的回答如下：")
            color = RANK_COLOR[0]
            st.markdown(f":{color}[{answer}]")
