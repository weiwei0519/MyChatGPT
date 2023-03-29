# coding=UTF-8
# 文本情感分类模型

'''
@File: reward_model
@Author: WeiWei
@Time: 2023/2/26
@Email: weiwei_519@outlook.com
@Software: PyCharm
'''
from pathlib import Path
import logging
import time, datetime
from random import sample
from transformers import BertConfig
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
from torch.optim import AdamW
from utils.dataset_util import TextClassifyDataset
from train.text_classify_model_train import model_train

# 训练参数
batch_size = 32
epochs = 100
learning_rate = 5e-5  # 学习率
context_length = 512
action = 'validate'  # train 训练   validate 测试  prod  生产运行
Imdb_path = '../datasets/IMDB_movie/aclImdb/'
pretrained_model_dir = "../models/distilbert-base-uncased/"
model_output_dir = "./models/Company-RewardModel0.1-Chinese/"
sample_rate = 0.01  # 样本抽样率，节省训练时间。

# device : GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
today = datetime.datetime.now().strftime("%Y-%m-%d")
logging.basicConfig(filename=f'./logs/classify/classify_{today}.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filemode='a'
                    )


def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir / label_dir).iterdir():
            texts.append(text_file.read_text(encoding='utf-8'))
            labels.append(0 if label_dir == "neg" else 1)
    return texts, labels


train_texts, train_labels = read_imdb_split(Imdb_path + 'train')
test_texts, test_labels = read_imdb_split(Imdb_path + 'test')
logging.info('load imdb data finished!')
logging.info('train_texts shape = {0} before sampling'.format(len(train_texts)))
logging.info('test_texts shape = {0} before sampling'.format(len(test_texts)))
sample_idx = sample([ids for ids in range(len(train_texts))], int(round(len(train_texts) * sample_rate, 0)))
train_texts = [train_texts[idx] for idx in sample_idx]
train_labels = [train_labels[idx] for idx in sample_idx]
sample_idx = sample([ids for ids in range(len(test_texts))], int(round(len(test_texts) * sample_rate, 0)))
test_texts = [test_texts[idx] for idx in sample_idx]
test_labels = [test_labels[idx] for idx in sample_idx]
train_max_len = max([len(s) for s in train_texts])
test_max_len = max([len(s) for s in test_texts])
logging.info('train_texts shape = {0} after sampling'.format(len(train_texts)))
logging.info('test_texts shape = {0} after sampling'.format(len(test_texts)))

#  进行分词，转为一串 id，定义分词器，并构造数据集
logging.info("Tokenizing train, validate, test text ")
tokenizer = DistilBertTokenizerFast.from_pretrained(pretrained_model_dir)
config = BertConfig.from_pretrained(pretrained_model_dir)
# train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=context_length)
# test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=context_length)

logging.info("create test classify Datasets ")
train_dataset = TextClassifyDataset(tokenizer=tokenizer,
                                    texts=train_texts,
                                    labels=train_labels,
                                    max_len=context_length,
                                    )
test_dataset = TextClassifyDataset(tokenizer=tokenizer,
                                   texts=test_texts,
                                   labels=test_labels,
                                   max_len=context_length,
                                   )
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
del train_texts, train_labels, test_texts, test_labels

# 定义一个用于预测得分的模型，以及优化器：
logging.info("Loading blank bert model ")
model = DistilBertForSequenceClassification(config)
model.to(device)
model_size = sum(t.numel() for t in model.parameters())
logging.info(f"model size: {model_size / 1000 ** 2:.1f}M parameters")
model.train()
optimizer = AdamW(model.parameters(), lr=learning_rate)

# 模型训练
pre = time.time()
current = time.strftime("%Y-%m-%d %H:%M:%S")
model.train()
logging.info('reward model training start on {0}'.format(current))
model_train(
    tokenizer=tokenizer,
    model=model,
    train_dataset=train_dataset,
    val_dataset=test_dataset,
    batch_size=batch_size,
    epochs=epochs,
    learning_rate=learning_rate,
    device=device,
    model_dir=model_output_dir,
)
# for epoch in range(3):
#     for (b_ix, batch) in enumerate(train_loader):
#         optimizer.zero_grad()
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['labels'].to(device)
#         outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#         loss = outputs[0]
#         loss.backward()
#         optimizer.step()
logging.info('训练时间：', time.time() - pre)
