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
logger = logging.getLogger()

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

# if model_glm is None or tokenizer_glm is None:
#     logging.info('加载ChatGLM预训练模型')
#     tokenizer_glm = AutoTokenizer.from_pretrained(MODEL_CONFIG['model_glm'], trust_remote_code=True)
#     config_glm = AutoConfig.from_pretrained(MODEL_CONFIG['model_glm'], trust_remote_code=True)
#     # model_glm = AutoModel.from_pretrained(MODEL_CONFIG['model_glm'],
#     #                                       trust_remote_code=True,
#     #                                       ignore_mismatched_sizes=True).float()  # CPU
#     model_glm = AutoModel.from_pretrained(MODEL_CONFIG['model_glm'],
#                                           trust_remote_code=True,
#                                           ignore_mismatched_sizes=True).half().cuda()  # GPU
#     logging.info('ChatGLM预训练模型加载成功')

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
def generate_text(prompt, do_sample):
    """
    模型生成文字。
    """
    # 加载预训练模型：
    logger.info('加载AIA预训练模型')
    tokenizer = T5Tokenizer.from_pretrained(MODEL_CONFIG['model_AIA'])
    config = T5Config.from_pretrained(MODEL_CONFIG['model_AIA'])
    model = T5ForConditionalGeneration.from_pretrained(MODEL_CONFIG['model_AIA'])
    model.to(GPU)
    logger.info('AIA预训练模型加载成功')

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
        logger.info(f'action: {action}')
    else:
        # 问答模式
        action = '问答模式'
        return_seqs = 1
        logger.info(f'action: {action}')

    try:
        logger.info(f'使用预训练模型：AIA-GPT prompt: {prompt}')
        answers = infer_answer(text=prompt,
                               tokenizer=tokenizer,
                               model=model,
                               # do_sample=do_sample,
                               samples=return_seqs,
                               out_length=MODEL_CONFIG['max_gen_seq_len'],
                               )
        answers = [answer.replace('\n', '<br>').replace('\t', '   ') for answer in answers]
        logger.info(f'answer in generator: {len(answers)} x {len(answers[0])}')
        logger.info(f'answer in generator: {answers}')
        if action == '标注模式':
            st.session_state['answer_to_rank'] = {}
            for _, answer in enumerate(answers):
                st.session_state['answer_to_rank'][answer] = '-1'  # -1 为初始rank，等于未排序
        elif action == '问答模式':
            st.session_state['answer'] = answers[0]
        del model, tokenizer, config
        torch.cuda.empty_cache()
    except NameError:
        answers = ['模型加载失败！请联系系统管理员']


def generate_from_GPT4(prompt):
    logger.info(f'GPT4 prompt: {prompt}')
    generator = st.session_state['GPT4_client']
    task = generator.generate(text=prompt)
    task.wait()
    response = task.output.blocks[0].text
    # print(f'GPT4 answer is: {response}')
    st.session_state['program_answer'] = response


def generate_from_ChatGLM(prompt):
    logger.info('加载ChatGLM预训练模型')
    tokenizer_glm = AutoTokenizer.from_pretrained(MODEL_CONFIG['model_glm'], trust_remote_code=True)
    config_glm = AutoConfig.from_pretrained(MODEL_CONFIG['model_glm'], trust_remote_code=True)
    # model_glm = AutoModel.from_pretrained(MODEL_CONFIG['model_glm'],
    #                                       trust_remote_code=True,
    #                                       ignore_mismatched_sizes=True).float()  # CPU
    model_glm = AutoModel.from_pretrained(MODEL_CONFIG['model_glm'],
                                          trust_remote_code=True,
                                          ignore_mismatched_sizes=True).half().cuda()  # GPU
    logger.info('ChatGLM预训练模型加载成功')
    logger.info(f'ChatGLM prompt: {prompt}')
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

prompt_tab, label_tab, program_tab = st.tabs(['Prompt', 'Label', 'Program'])
answer_to_rank = []
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: left;} </style>',
         unsafe_allow_html=True)

# st.write('<style>div.st-bf{flex-direction:row;} div.st-ag{font-weight:bold;padding-left:2px;}</style>',
#          unsafe_allow_html=True)

######################### 页面定义区（问答页面） ########################
with prompt_tab:
    st.subheader(':blue[AIA-GPT对话问答]')
    with st.expander('', expanded=True):  # expanded 是展开还是收缩状态
        qa_query_txt = st.text_area('🔍 请输入您的提问：', placeholder='此处输入您的提问', key='qa_query')
        if qa_query_txt != '' and st.session_state['prompt'] != qa_query_txt:
            st.session_state['prompt'] = qa_query_txt
            generate_text(st.session_state['prompt'], do_sample=False)

    with st.expander('💡 GPT的回答如下：', expanded=True):
        if 'answer' in st.session_state:
            answer = st.session_state['answer']
            if answer != '':
                color = RANK_COLOR[0]
                st.markdown(f":{color}[{answer}]", unsafe_allow_html=True)

######################### 页面定义区（标注页面） ########################
with label_tab:
    st.subheader(':blue[AIA-GPT对话问答]')
    with st.expander('', expanded=True):  # expanded 是展开还是收缩状态
        label_query_txt = st.text_area('🔍 请输入您的提问：', placeholder='此处输入您的提问', key='label_query')
        if label_query_txt != '' and st.session_state['query'] != label_query_txt:
            st.session_state['query'] = label_query_txt
            generate_text(st.session_state['query'], do_sample=True)

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
        file.seek(0)
        file.truncate()
        if len(content) != 0 and content != '':
            rank_pairs = json.loads(content)
        else:
            rank_pairs = {}
        length = len(rank_pairs)
        logger.info(f'orignal json length: {length}')
        rank_pairs[length] = {}
        rank_pairs[length]['prompt'] = st.session_state['prompt'].strip()
        ranking = {}
        for ans, rank in answer_to_rank.items():
            ranking[rank] = ans
        ranked_answers = [ranking[key] for key in sorted(ranking.keys())]
        rank_pairs[length]['ranked_answers'] = ranked_answers
        # dumps()：将dict数据转化成json数据；   dump()：将dict数据转化成json数据后写入json文件
        content = json.dumps(rank_pairs, ensure_ascii=False, indent=2)
        logger.info(f'file save content length {len(content)}')
        # json.dump(content, file, ensure_ascii=False, indent=2)
        file.write(content)
        file.flush()
        st.success('保存成功，请更换prompt生成新的答案~', icon="✅")
        st.session_state.clear()
        # label_tab.empty()
        # st._rerun()
        # st_autorefresh(interval=2000, limit=100, key="fizzbuzzcounter")
#
# ######################### 页面定义区（代码辅助工具页面） ########################
# with program_tab:
#     st.subheader(':blue[提供面向IT开发人员的代码辅助工具]')
#     with st.expander('', expanded=True):  # expanded 是展开还是收缩状态
#         func_options = ('请选择：', '生成代码', '代码解读', '修复bug', '单元测试', '添加注释')
#         # gene_tab, read_tab, bug_tab, test_tab, comment_tab = st.tabs(func_options)
#         func_radio = st.radio(label='请选择您的需求：',
#                               options=func_options,
#                               index=0,
#                               key='func_options',
#                               )
#         lang_options = ('Java', 'SQL', 'Vue.js', 'Python', 'C++', 'Cobol', 'Swift')
#         lang_radio = st.radio(label='请选择您代码的语言：',
#                               options=lang_options,
#                               index=0,
#                               key='lang_options_gene',
#                               )
#         prompt = ""
#         if func_radio == '生成代码':
#             prompt = "请帮我生成一段" + lang_radio + "代码，需求是：\n"
#         elif func_radio == '代码解读':
#             prompt = "请帮我总结如下" + lang_radio + "代码的用途，代码是：\n"
#         elif func_radio == '修复bug':
#             prompt = "请帮我发现并修复如下" + lang_radio + "代码中存在的问题，代码是：\n"
#         elif func_radio == '单元测试':
#             prompt = "请帮我生成如下" + lang_radio + "代码的单元测试，代码是：\n"
#         elif func_radio == '添加注释':
#             prompt = "请帮我为如下" + lang_radio + "代码添加注释，代码是：\n"
#         program_request = st.text_area(label='🔍 选择哪种程序语言后，请再输入您的详细需求：',
#                                        placeholder='此处输入您的详细需求',
#                                        value=prompt,
#                                        height=200,
#                                        key='program_prompt')
#         ask = st.button('提问')
#         if ask and program_request != '':
#             last_program_prompt = st.session_state['program_prompt']
#             # st.session_state['program_prompt'] = program_request
#             # generate_from_GPT4(program_request)
#             st.session_state.disabled = True  # 防止重复提交，节省内存
#             generate_from_ChatGLM(program_request)
#
#         if 'program_answer' in st.session_state and st.session_state['program_answer'] != '':
#             answer = st.session_state['program_answer']
#             st.text("💡 GPT的回答如下：")
#             color = RANK_COLOR[0]
#             st.markdown(f":{color}[{answer}]")
#
#         st.session_state.disabled = False
