# coding=UTF-8
# prompt model and verification

'''
@File: prompt_model
@Author: WeiWei
@Time: 2023/3/2
@Email: weiwei_519@outlook.com
@Software: PyCharm
'''

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from utils.dataset_util import preprocess, postprocess

# device : GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
action = 'validate'  # train 训练   validate 测试  prod  生产运行
model_dir = "./models/CompanyModel0.1-Prompt-Chinese/"
# 模型参数
context_length = 512


def infer_answer(model, tokenizer, text, sample=True, top_p=1, temperature=0.7):
    '''sample：是否抽样。生成任务，可以设置为True;
    top_p：0-1之间，生成的内容越多样'''
    text = preprocess(text)
    encoding = tokenizer(text=[text],
                         truncation=True,
                         pad_to_max_length=True,
                         padding='max_length',
                         max_length=context_length,
                         return_tensors="pt").to(device)

    if not sample:
        out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False,
                             max_new_tokens=512,
                             num_beams=1, length_penalty=0.6)
    else:
        out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False,
                             max_new_tokens=512,
                             do_sample=True, top_p=top_p, temperature=temperature,
                             no_repeat_ngram_size=3)
    out_text = tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)
    return postprocess(out_text[0])


if __name__ == '__main__':
    # 加载Company预训练模型
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    config = T5Config.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    model.to(device)
    cont = True
    while cont:
        input_text = str(input("请输入/Please input： "))

        if input_text == "exit":
            cont = False
        else:
            output_text = infer_answer(model=model, tokenizer=tokenizer, text=input_text)
            print(output_text)
