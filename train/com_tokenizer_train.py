# coding=UTF-8
# 基于自定义数据集，训练tokenizer

'''
@File: com_tokenizer_train
@Author: WeiWei
@Time: 2023/2/25
@Email: weiwei_519@outlook.com
@Software: PyCharm
'''

import torch
from tokenizers.implementations import BertWordPieceTokenizer
from tokenizers.processors import BertProcessing

# device : GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

train_file = "../datasets/company_datasets/doc_dataset.txt"  # 训练文本文件
vocab_file = "../models/CompanyModel0.1-ChatGPT-Chinese/"  # tokenizer & vocab输出目录
vocab_size = 20000
min_frequency = 2
limit_alphabet = 20000
special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]  # 适用于Bert和Albert

# Initialize a tokenizer
tokenizer = BertWordPieceTokenizer(
    clean_text=True, handle_chinese_chars=True, strip_accents=True, lowercase=True,
)

# Customize training
tokenizer.train(
    train_file,
    vocab_size=vocab_size,
    min_frequency=min_frequency,
    show_progress=True,
    special_tokens=special_tokens,
    limit_alphabet=limit_alphabet,
    wordpieces_prefix="##"
)
# 保存tokenizer
tokenizer.save(vocab_file + "vocab.txt")

# 分词结果测试
tokenizer = BertWordPieceTokenizer.from_file(vocab_file + "vocab.txt")

tokenizer._tokenizer.post_processor = BertProcessing(
    ("[CLS]", tokenizer.token_to_id("[SEP]")),
    ("[SEP]", tokenizer.token_to_id("[CLS]")),
)
tokenizer.enable_truncation(max_length=512)

tokenizer.encode("盛世经典尊享版终身寿险").prompt_tokens
