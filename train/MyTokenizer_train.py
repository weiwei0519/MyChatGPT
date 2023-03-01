# coding=utf-8
# 基于自己的数据集训练tokenizer
"""
# Project Name: MyGPT
# File Name: MyTokenizer_train
# Author: NSNP577
# Creation at 2023/2/23 11:38
# IDE: PyCharm
# Describe: 
"""

import torch
import logging
import os
import docx2txt
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, NFKC, Sequence
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

# device : GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
logging.basicConfig(level=logging.INFO)

# 训练参数
batch_size = 128
epochs = 50
learning_rate = 1e-2  # 学习率
max_len = 512
action = 'train'  # train 训练   validate 测试  prod  生产运行
model_dir = "../models/CompanyModel0.1-GPT2-Chinese/"


# 实例化BPE tokenizer
tokenizer = Tokenizer(BPE())

# 规范化操作包括 lower-casing 和 unicode-normalization
tokenizer.normalizer = Sequence([
    NFKC(),
    Lowercase()
])

# pre-tokenizer 以空白作为词语边界
tokenizer.pre_tokenizer = Whitespace()

# And finally, let's plug a decoder so we can recover from a tokenized input to the original one
tokenizer.decoder = ByteLevelDecoder()

# 实例化tokenizer训练实例
trainer = BpeTrainer(vocab_size=25000, show_progress=True)
doc_path = '../datasets/document/'
files = os.listdir(doc_path)
doc_texts = []
for file in files:
    # f = open(doc_path + file, 'r', encoding='UTF-8')
    text = docx2txt.process(doc_path + file)
    doc_texts.append(text.replace("\n\n", "\n"))
tokenizer.train_from_iterator(iterator=iter(doc_texts), trainer=trainer)

print("Trained vocab size: {}".format(tokenizer.get_vocab_size()))

# You will see the generated files in the output.
tokenizer.model.save(model_dir)
enc = tokenizer.encode('友童乐齿医疗保险合同内容变更')
print(tokenizer.encode('友童乐齿医疗保险合同内容变更'))