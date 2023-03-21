# coding=UTF-8
# mini-chatgpt

'''
@File: MyChatGPT
@Author: WeiWei
@Time: 2023/2/18
@Email: weiwei_519@outlook.com
@Software: PyCharm
'''

from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextGenerationPipeline, AutoTokenizer, AutoConfig
import torch
import logging
import random
from utils.dataset_util import preprocess, postprocess
from utils.NLP import language, is_company_prompt
# import gradio as gr

print(torch.version.cuda)

# device : GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
logging.basicConfig(level=logging.INFO)
ui = 'command'  # chatbot聊天网页模式，command命令行模式

# 加载中文对话模型
# 微信搜索小程序“元语智能”
# ChatYuan: 元语功能型对话大模型
# 这个模型可以用于问答、结合上下文做对话、做各种生成任务，包括创意性写作，也能回答一些像法律、新冠等领域问题。
# 它基于PromptCLUE-large结合数亿条功能对话多轮对话数据进一步训练得到。
# PromptCLUE-large:在1000亿token中文语料上预训练，累计学习1.5万亿中文token，并且在数百种任务上进行Prompt任务式训练。
# 针对理解类任务，如分类、情感分析、抽取等，可以自定义标签体系；针对多种生成任务，可以进行采样自由生成。
# input_text0 = "帮我写一个请假条，我因为新冠不舒服，需要请假3天，请领导批准"
# input_text1 = "你能干什么"
# input_text2 = "用英文写一封道歉的邮件，表达因为物流延误，不能如期到达，我们可以赔偿贵公司所有损失"
# input_text3 = "写一个文章，题目是未来城市"
# input_text4 = "写一个诗歌，关于冬天"
# input_text5 = "从南京到上海的路线"
# input_text6 = "学前教育专业岗位实习中，在学生方面会存在问题，请提出改进措施。800字"
# input_text7 = "根据标题生成文章：标题：屈臣氏里的化妆品到底怎么样？正文：化妆品，要讲究科学运用，合理搭配。屈臣氏起码是正品连锁店。请继续后面的文字。"
# input_text8 = "帮我对比几款GPU，列出详细参数对比，并且给出最终结论"
tokenizer_cn_chatgpt = T5Tokenizer.from_pretrained("./models/ChatYuan-large-v1")
model_cn_chatgpt = T5ForConditionalGeneration.from_pretrained("./models/ChatYuan-large-v1")
model_cn_chatgpt.to(device)
sample = True
context_length = 512
# 加载英文GPT2预训练模型
# tokenizer_en_gpt2 = GPT2Tokenizer.from_pretrained('./models/OpenAI-GPT2-XLarge')
# model_en_gpt2 = GPT2LMHeadModel.from_pretrained('./models/OpenAI-GPT2-XLarge')
# text_generator_en_gpt2 = TextGenerationPipeline(model_en_gpt2, tokenizer_en_gpt2)
# model_en_gpt2.to(device)
# max_length = 100
# 加载公司GPT2训练模型
# pretrained_model_dir = "./models/bert-base-chinese/"
# model_output_dir = "./models/CompanyModel0.1-GPT2-Chinese/"
# context_length = 512
# tokenizer_company = AutoTokenizer.from_pretrained(pretrained_model_dir)
# model_company = GPT2LMHeadModel.from_pretrained(model_output_dir,
#                                         bos_token_id=tokenizer_company.bos_token_id,
#                                         eos_token_id=tokenizer_company.eos_token_id,
#                                         pad_token_id=tokenizer_company.eos_token_id)
# model_company.to(device)

def infer_answer(text,
                 model=model_cn_chatgpt,
                 tokenizer=tokenizer_cn_chatgpt,
                 ):
    text = preprocess(text)
    # encode context the generation is conditioned on
    encoding = tokenizer_cn_chatgpt(text=[text],
                                    truncation=True,
                                    pad_to_max_length=True,
                                    padding='max_length',
                                    max_length=context_length,
                                    return_tensors="pt").to(device)
    # set no_repeat_ngram_size to 2
    # top_p：0 - 1 之间，生成的内容越多样
    if not sample:
        out = model_cn_chatgpt.generate(**encoding, return_dict_in_generate=True, output_scores=False,
                                        max_new_tokens=512,
                                        num_beams=1, length_penalty=0.6)
    else:
        out = model_cn_chatgpt.generate(**encoding, return_dict_in_generate=True, output_scores=False,
                                        max_new_tokens=512,
                                        do_sample=True, top_p=1, temperature=0.7,
                                        no_repeat_ngram_size=3)
    out_text = tokenizer_cn_chatgpt.batch_decode(out["sequences"], skip_special_tokens=True)
    return "".join(postprocess(out_text[0]).split())  # 去除空格，恢复制表符/格式符


def predict(input, history=[]):
    respose = infer_answer(input)
    res = [(input, respose)]
    return res, history


def add_text(state, text):
    res = infer_answer(text)
    state = state + [(text, res)]
    return state, state


# 但这样有时可能会出现问题，例如模型陷入一个循环，不断生成同一个单词。
# 为了避免这种情况， GPT-2 设置了一个 top-k 参数，这样模型就会从概率前 k 大的单词中随机选取一个单词，作为下一个单词。
def select_top_k(predictions, k=10):
    predicted_tokens = random.choice(
        predictions[0, -1, :].sort(descending=True)[1][:10]).item()
    return predicted_tokens


if __name__ == '__main__':
    if ui == 'chatbot':
        # 网页模式
        with gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}") as demo:
            chatbot = gr.Chatbot(elem_id="chatbot")
            state = gr.State([])

            with gr.Row():
                txt = gr.Textbox(show_label=False, placeholder="输入文本").style(container=False)

            txt.submit(add_text, [state, txt], [state, chatbot])

        demo.launch(server_name='0.0.0.0', share=True)
    elif ui == 'command':
        # 命令行模式
        cont = True
        while cont:
            input_text = str(input("请输入/Please input： "))

            if input_text == "exit":
                cont = False
            else:
                # if is_company_prompt(input_text):
                #     output_text = answer(model=model_company, tokenizer=tokenizer_company, text=input_text)
                #     print("".join(output_text.split()))
                #     continue
                lang = language(input_text)
                # intent = intent_judge(input_text)
                if lang == 'cn':
                    output_text = infer_answer(text=input_text, model=model_cn_chatgpt, tokenizer=tokenizer_cn_chatgpt)
                    print(output_text)
                elif lang == 'en':
                    # output = text_generator_en_gpt2(input_text, max_length=max_length, do_sample=True)[0]
                    # output_text = output['generated_text']
                    # print(output_text)
                    print('暂不支持英文')
