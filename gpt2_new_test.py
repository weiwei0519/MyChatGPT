# coding=utf-8
"""
# Project Name: MyChatGPT
# File Name: gpt2_new_test
# Author: NSNP577
# Creation at 2023/3/18 17:37
# IDE: PyCharm
# Describe: 
"""

import torch, os, re, pandas as pd, json
from sklearn.model_selection import train_test_split
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding, GPT2Tokenizer, GPT2LMHeadModel, \
    AutoConfig
from transformers import Trainer, TrainingArguments, AutoConfig
from datasets import Dataset

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def pretty_print(text, max_len_line=100):
    words = text.split(' ')
    len_line = 0
    line = ''
    for w in words:
        if w == '\n':
            print(line)
            line = ''
            continue
        if (len(line) + len(w)) > max_len_line:
            print(line)
            line = ''
        line += ' ' + w
    print(line)


# 加载模型
base_model = GPT2LMHeadModel.from_pretrained('./models/gpt2')
base_tokenizer = GPT2Tokenizer.from_pretrained('./models/gpt2')

# 模型加载测试
text = "I work as a data scientist"
text_ids = base_tokenizer.encode(text, return_tensors='pt')
generated_text_samples = base_model.generate(text_ids,
                                             max_length=50,
                                             num_beams=5,
                                             no_repeat_ngram_size=2,
                                             num_return_sequences=5,
                                             early_stopping=True)
for i, beam in enumerate(generated_text_samples):
    print(f"{i}: {base_tokenizer.decode(beam, skip_special_tokens=True)}")
    print()


# 使用自有数据集，对模型进行微调
# the eos and bos tokens are defined
bos = '<|endoftext|>'
eos = '<|EOS|>'
pad = '<|pad|>'

special_tokens_dict = {'eos_token': eos, 'bos_token': bos, 'pad_token': pad}

# the new token is added to the tokenizer
num_added_toks = base_tokenizer.add_special_tokens(special_tokens_dict)

# the model config to which we add the special tokens
config = AutoConfig.from_pretrained('./models/gpt2',
                                    bos_token_id=base_tokenizer.bos_token_id,
                                    eos_token_id=base_tokenizer.eos_token_id,
                                    pad_token_id=base_tokenizer.pad_token_id,
                                    output_hidden_states=False)

# the pre-trained model is loaded with the custom configuration
base_model = GPT2LMHeadModel.from_pretrained('./models/gpt2', config=config)

# the model embedding is resized
base_model.resize_token_embeddings(len(base_tokenizer))