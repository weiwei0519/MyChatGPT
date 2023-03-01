# coding=UTF-8
# chatbot UI

'''
@File: chatbot.py
@Author: WeiWei
@Time: 2023/2/18
@Email: weiwei_519@outlook.com
@Software: PyCharm
'''

import gradio as gr
from transformers import GPT2LMHeadModel, AutoTokenizer
from utils.dataset_util import preprocess, postprocess
import torch

pretrained_model_dir = "./models/bert-base-chinese/"
model_output_dir = "./models/CompanyModel0.1-GPT2-Chinese/"
context_length = 512
sample = True  # 是否抽样。生成任务，可以设置为True;

# device : GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 加载预训练模型：
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir)
model = GPT2LMHeadModel.from_pretrained(model_output_dir,
                                        bos_token_id=tokenizer.bos_token_id,
                                        eos_token_id=tokenizer.eos_token_id,
                                        pad_token_id=tokenizer.eos_token_id)
model.to(device)


def infer_model(text: str) -> str:
    text = preprocess(text)
    # encode context the generation is conditioned on
    encoding = tokenizer(text=[text],
                         truncation=True,
                         pad_to_max_length=True,
                         padding='max_length',
                         max_length=context_length,
                         return_tensors="pt").to(device)
    # set no_repeat_ngram_size to 2
    # top_p：0 - 1 之间，生成的内容越多样
    if not sample:
        out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False,
                             max_new_tokens=512,
                             num_beams=1, length_penalty=0.6)
    else:
        out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False,
                             max_new_tokens=512,
                             do_sample=True, top_p=1, temperature=0.7,
                             no_repeat_ngram_size=3)
    out_text = tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)
    return "".join(postprocess(out_text[0]).split())  # 去除空格，恢复制表符/格式符


def predict(input, history=[]):
    respose = infer_model(input)
    res = [(input, respose)]
    return res, history


def add_text(state, text):
    res = infer_model(text)
    state = state + [(text, res)]
    return state, state


if __name__ == "__main__":
    with gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}") as demo:
        chatbot = gr.Chatbot(elem_id="chatbot")
        state = gr.State([])

        with gr.Row():
            txt = gr.Textbox(show_label=False, placeholder="输入文本").style(container=False)

        txt.submit(add_text, [state, txt], [state, chatbot])

    demo.launch(server_name='0.0.0.0', share=True)
