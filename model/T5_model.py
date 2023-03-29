# coding=utf-8
# T5 (Text-To-Text transfer transformer) model definition
"""
# Project Name: MyGPT
# File Name: T5_model
# Author: NSNP577
# Creation at 2023/3/1 9:35
# IDE: PyCharm
# Describe: 
"""
import sys
sys.path.append("../")

import torch
from torch import nn
from torch.nn import Identity
from utils.dataset_util import InputOutputDataset, preprocess, postprocess
from torch.utils.data import DataLoader
import logging
import os
import random
import time
import datetime
from tqdm import tqdm
import io
import glob
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, T5PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from train.T5_model_train import model_train
from utils.gpu_track import MemTracker
import inspect
from dataclasses import dataclass
from typing import Optional, Tuple

# device : GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
today = datetime.datetime.now().strftime("%Y-%m-%d")
logging.basicConfig(filename=f'./logs/T5/T5_{today}.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filemode='a'
                    )
logging.info(device)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"  # 防止GPU内存溢出
# 追踪GPU Mem的消耗情况。
frame = inspect.currentframe()  # define a frame to track
gpu_tracker = MemTracker(frame)

# 训练参数
batch_size = 2
epochs = 10000
learning_rate = 1e-5  # 学习率
text_length = 50
input_ids_length = 400
action = 'train'  # train 训练    validate 测试     prod 生产运行   checkpoint 继续训练     fine-tuning 微调模型
pretrained_model_dir = "../models/ChatYuan-large-v1/"
model_output_dir = "../models/chatgpt-aia-chinese/ttt-aia-chinese"


@dataclass
class CausalLMOutputWithCrossAttentions(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    value: Optional[torch.FloatTensor] = None


class T5ModelWithValueModel(T5PreTrainedModel):
    """The T5ModelWithValueModel class implements a TTT language model with a secondary, scalar head."""

    def __init__(self, config):
        super().__init__(config)
        config.num_labels = 1
        self.transformer = T5ForConditionalGeneration(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.v_head = ValueHead(config)
        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def detach_value_head(self):
        self.v_head.detach_head = True

    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            mc_token_ids=None,
            lm_labels=None,
            mc_labels=None,
            return_dict=False,
            output_attentions=False,
            output_hidden_states=False,
            use_cache=True,
    ):
        loss = None
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
        )
        hidden_states = transformer_outputs[0]  # (batch, seq_len, 768)
        lm_logits = self.lm_head(hidden_states)  # (batch, seq_len, vocab_size)
        value = self.v_head(hidden_states).squeeze(-1)  # (batch, seq_len)

        if not return_dict:
            outputs = (lm_logits, loss, value,)
            return outputs

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
            value=value,
        )

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }


class ValueHead(nn.Module):
    """The ValueHead class implements a head for GPT2 that returns a scalar for each output token."""

    def __init__(self, config):
        super().__init__()
        self.detach_head = False
        self.summary_type = config.summary_type if hasattr(config, "summary_type") else "last"
        if self.summary_type == "attn":
            raise NotImplementedError

        self.summary = Identity()
        if hasattr(config, "summary_use_proj") and config.summary_use_proj:
            if hasattr(config, "summary_proj_to_labels") and config.summary_proj_to_labels and config.num_labels > 0:
                num_classes = config.num_labels
            else:
                num_classes = config.hidden_size
            self.summary = nn.Linear(config.hidden_size, num_classes)

        self.activation = Identity()
        if hasattr(config, "summary_activation") and config.summary_activation == "tanh":
            self.activation = nn.Tanh()

        self.first_dropout = Identity()
        if hasattr(config, "summary_first_dropout") and config.summary_first_dropout > 0:
            self.first_dropout = nn.Dropout(config.summary_first_dropout)

        self.last_dropout = Identity()
        if hasattr(config, "summary_last_dropout") and config.summary_last_dropout > 0:
            self.last_dropout = nn.Dropout(config.summary_last_dropout)

        self.flatten = nn.Flatten()

    def forward(self, hidden_states, cls_index=None):
        if self.detach_head:
            output = hidden_states.detach()
        else:
            output = hidden_states
        output = self.first_dropout(output)
        output = self.summary(output)
        output = self.activation(output)
        output = self.last_dropout(output)

        return output


# 但这样有时可能会出现问题，例如模型陷入一个循环，不断生成同一个单词。
# 为了避免这种情况， GPT-2 设置了一个 top-k 参数，这样模型就会从概率前 k 大的单词中随机选取一个单词，作为下一个单词。
def select_top_k(predictions, k=10):
    predicted_tokens = random.choice(
        predictions[0, -1, :].sort(descending=True)[1][:10]).item()
    return predicted_tokens


def train():
    # gpu_tracker.track()
    global action
    # 加载aia预训练模型，若不存在则初始化空模型
    checkpoint = glob.glob(os.path.join(model_output_dir, 'checkpoint-*'))  # 按照目前trainer的训练输出，只会存在一个checkpoint
    if len(checkpoint) > 0 and action == 'checkpoint':
        # 从checkpoint断点继续训练
        checkpoint = (checkpoint[0]).replace("\\", "/")
        tokenizer = T5Tokenizer.from_pretrained(checkpoint)
        config = T5Config.from_pretrained(checkpoint)
        model = T5ForConditionalGeneration.from_pretrained(checkpoint)
        model.to(device)
    elif action == 'fine-tuning':
        tokenizer = T5Tokenizer.from_pretrained(model_output_dir)
        config = T5Config.from_pretrained(model_output_dir)
        # model = T5ForConditionalGeneration.from_pretrained(model_output_dir)
        model = T5ForConditionalGeneration(config)
        model.to(device)
    else:
        # 初始化预训练模型
        tokenizer = T5Tokenizer.from_pretrained(model_output_dir)
        config = T5Config.from_pretrained(model_output_dir)
        model = T5ForConditionalGeneration(config)
        model.to(device)
    model_size = sum(t.numel() for t in model.parameters())
    logging.info(f"model size: {model_size / 1000 ** 2:.1f}M parameters")
    # gpu_tracker.track()

    # 数据集
    file = '../datasets/company_datasets/aiacn/Prompt_Finetuning.xlsx'
    content = pd.read_excel(file)
    logging.info(content.head(5))
    text_pairs = content.iloc[:, [0, 1]]
    # lines = io.open(file, encoding='UTF-8').read().strip().split('\n')
    # texts_pairs = [[w for w in l.split('>>>')] for l in lines]
    # source_texts, target_texts = zip(*texts_pairs)
    source_texts = content.iloc[:, 0].values.tolist()
    target_texts = content.iloc[:, 1].values.tolist()
    src_max_len = max([len(text) for text in source_texts])
    tgt_max_len = max([len(text) for text in target_texts])
    train_set = InputOutputDataset(tokenizer=tokenizer,
                                   source_texts=source_texts,
                                   target_texts=target_texts,
                                   text_length=text_length,
                                   input_ids_length=input_ids_length,
                                   )
    logging.info('dataset''s shape = {0}, {1}'.format(train_set.source_shape, train_set.target_shape))

    # 开始模型训练
    pre = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 定义优化器

    model_train(
        tokenizer=tokenizer,
        model=model,
        dataset=train_set,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        action=action,
        model_dir=model_output_dir,
    )
    # trainer.evaluate()
    logging.info('训练时间：', time.time() - pre)


def infer_answer(model, tokenizer, text, out_length, do_sample=True, samples=1, top_p=1, temperature=0.7):
    '''sample：是否抽样。生成任务，可以设置为True;
    top_p：0-1之间，生成的内容越多样'''
    logging.info(f"input text: {text}")
    text = preprocess(text)
    encoding = tokenizer(text=[text],
                         truncation=True,
                         pad_to_max_length=True,
                         padding='max_length',
                         max_length=text_length,
                         return_tensors="pt").to(device)

    if not do_sample:
        out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False,
                             max_new_tokens=out_length,
                             num_beams=1, length_penalty=0.6)
    else:
        out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False,
                             max_new_tokens=out_length, do_sample=do_sample,
                             top_p=top_p, temperature=temperature,
                             num_return_sequences=samples,
                             no_repeat_ngram_size=3)
    out_text = tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)
    out_text = [postprocess(text) for text in out_text]
    logging.info(f'out_text {out_text}')
    return out_text


# 模型测试
def validate(model, tokenizer, input_text):
    total_predicted_text = infer_answer(input_text)

    return "".join(total_predicted_text.split())


if __name__ == '__main__':
    if action == 'train' or action == 'checkpoint' or action == 'fine-tuning':
        train()
    elif action == 'validate':
        # 加载Company预训练模型
        tokenizer = T5Tokenizer.from_pretrained(pretrained_model_dir)
        config = T5Config.from_pretrained(pretrained_model_dir)
        model = T5ForConditionalGeneration.from_pretrained(pretrained_model_dir)
        model.to(device)
        cont = True
        while cont:
            input_text = str(input("请输入/Please input： "))

            if input_text == "exit":
                cont = False
            else:
                output_text = infer_answer(model=model,
                                           tokenizer=tokenizer,
                                           text=input_text,
                                           out_length=input_ids_length,
                                           samples=1,
                                           )
                print(output_text)

# input_text0 = "帮我写一个请假条，我因为新冠不舒服，需要请假3天，请领导批准"
# input_text1 = "你能干什么"
# input_text2 = "用英文写一封道歉的邮件，表达因为物流延误，不能如期到达，我们可以赔偿贵公司所有损失"
# input_text3 = "写一个文章，题目是未来城市"
# input_text4 = "写一个诗歌，关于冬天"
# input_text5 = "从南京到上海的路线"
# input_text6 = "学前教育专业岗位实习中，在学生方面会存在问题，请提出改进措施。800字"
# input_text7 = "根据标题生成文章：标题：屈臣氏里的化妆品到底怎么样？正文：化妆品，要讲究科学运用，合理搭配。屈臣氏起码是正品连锁店。请继续后面的文字。"
# input_text8 = "帮我对比几款GPU，列出详细参数对比，并且给出最终结论"
