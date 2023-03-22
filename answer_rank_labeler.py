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
from transformers import AutoTokenizer, GPT2LMHeadModel, TextGenerationPipeline
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from model.T5_model import infer_answer
# from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="AIA-GPT 管理平台",
    page_icon="🦈",
    layout="wide"
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODEL_CONFIG = {
    'model_name': './models/ChatYuan-large-v1',  # backbone
    'dataset_file': './datasets/company_datasets/human_rank_pairs.json',  # 标注数据集的存放文件
    'rank_list_len': 3,  # 排序列表的长度
    'max_gen_seq_len': 512,  # 生成答案最大长度
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
tokenizer = T5Tokenizer.from_pretrained(MODEL_CONFIG['model_name'])
config = T5Config.from_pretrained(MODEL_CONFIG['model_name'])
model = T5ForConditionalGeneration.from_pretrained(MODEL_CONFIG['model_name'])
model.to(device)

######################## 会话缓存初始化 ###########################
print('request & response!')

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


######################### 函数定义区 ##############################
def generate_text(prompt, do_sample):
    """
    模型生成文字。
    """
    print(f'prompt: {prompt}')
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
    else:
        # 问答模式
        action = '问答模式'
        return_seqs = 1
    answers = infer_answer(text=prompt,
                           tokenizer=tokenizer,
                           model=model,
                           # do_sample=do_sample,
                           samples=return_seqs
                           )
    print(f'answer in generator: {len(answers)} x {len(answers[0])}')
    if action == '标注模式':
        st.session_state['answer_to_rank'] = {}
        for _, answer in enumerate(answers):
            st.session_state['answer_to_rank'][answer] = '-1'  # -1 为初始rank，等于未排序
    elif action == '问答模式':
        st.session_state['answer'] = answers[0]


######################### 页面定义区（侧边栏） ########################
with st.sidebar:
    st.image(Image.open('./images/AIA_logo.png'))
    st.sidebar.title('📌 AIA-GPT 管理平台')
    st.sidebar.markdown('''
        ```python
        提供生成式提问测试，以及内容的排序标注
        
        Label Tab：标注模式。
            用户输入提示问题，系统输出多个答案。
            用户根据答案的准确性进行排序。
            
        Prompt Tab：问答模式。
            用户输入提示问题，系统输出唯一答案。
            
        ```
    ''')

label_tab, prompt_tab = st.tabs(['Label', 'Prompt'])
answer_to_rank = []

######################### 页面定义区（标注页面） ########################
with label_tab:
    with st.expander('AIA-GPT对话问答专区', expanded=True):  # expanded 是展开还是收缩状态
        label_query_txt = st.text_area('🔍 请输入您的提问：', placeholder='此处输入您的提问', key='label_query')
        if label_query_txt != '' and st.session_state['query'] != label_query_txt:
            st.session_state['query'] = label_query_txt
            generate_text(st.session_state['query'], do_sample=True)

    with st.expander('GPT的回答如下：', expanded=True):
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
                            st.markdown(f":{color}[{answer}]")
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
                    # color = RANK_COLOR[0]
                    # st.markdown(f":{color}[{input_answer}]")
                # st.session_state['answer_to_rank'] = answer_to_rank

        else:
            rank_col, answer_col = st.columns([1, 10])
            with rank_col:
                st.text("🥇 Choose Rank")
            with answer_col:
                st.text("💡 Generate Answers")

    answer_to_rank_new = st.session_state['answer_to_rank']
    print(f'answer_to_rank: {answer_to_rank_new}')

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

        # with open(MODEL_CONFIG['dataset_file'], 'a', encoding='utf8') as f:
        #     rank_texts = []
        #     for i in range(len(rank_results)):
        #         rank_texts.append(st.session_state['current_results'][rank_results.index(i + 1)])
        #     line = '\t'.join(rank_texts)
        #     f.write(f'{line}\n')
        file = open(MODEL_CONFIG['dataset_file'], 'r+', encoding='utf8')
        print(f'file information: {file}')
        content = file.read()
        print(f'orignal file content: {content}')
        file.seek(0)
        file.truncate()
        if len(content) != 0 and content != '':
            rank_pairs = json.loads(content)
        else:
            rank_pairs = {}
        length = len(rank_pairs)
        print(f'orignal json length: {length}')
        rank_pairs[length] = {}
        rank_pairs[length]['prompt'] = st.session_state['prompt'].strip()
        ranking = {}
        for ans, rank in answer_to_rank.items():
            ranking[rank] = ans
        ranked_answers = [ranking[key] for key in sorted(ranking.keys())]
        rank_pairs[length]['ranked_answers'] = ranked_answers
        # dumps()：将dict数据转化成json数据；   dump()：将dict数据转化成json数据后写入json文件
        content = json.dumps(rank_pairs, ensure_ascii=False, indent=2)
        print(f'file save content {content}')
        # json.dump(content, file, ensure_ascii=False, indent=2)
        file.write(content)
        file.flush()
        st.success('保存成功，请更换prompt生成新的答案~', icon="✅")
        st.session_state.clear()
        # label_tab.empty()
        # st._rerun()
        # st_autorefresh(interval=2000, limit=100, key="fizzbuzzcounter")

######################### 页面定义区（问答页面） ########################
with prompt_tab:
    with st.expander('AIA-GPT对话问答专区', expanded=True):  # expanded 是展开还是收缩状态
        qa_query_txt = st.text_area('🔍 请输入您的提问：', placeholder='此处输入您的提问', key='qa_query')
        if qa_query_txt != '' and st.session_state['prompt'] != qa_query_txt:
            st.session_state['prompt'] = qa_query_txt
            generate_text(st.session_state['prompt'], do_sample=False)

    with st.expander('GPT的回答如下：', expanded=True):
        if 'answer' in st.session_state:
            answer = st.session_state['answer']
            st.text("💡 Generated Answers")
            if answer != '':
                color = RANK_COLOR[0]
                st.markdown(f":{color}[{answer}]")
