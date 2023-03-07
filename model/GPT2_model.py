# coding=utf-8
# GPT2模型结构
"""
# Project Name: MyGPT
# File Name: GPT2_model
# Author: NSNP577
# Creation at 2023/2/23 12:39
# IDE: PyCharm
# Describe: 
"""

import torch
from utils.dataset_util import GeneDataset, preprocess, postprocess
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Identity
from torch import softmax
import logging
import os
import random
import time
from tqdm import tqdm
import io
from transformers import GPT2LMHeadModel, TextGenerationPipeline, GPT2Tokenizer, AutoTokenizer, AutoConfig
from transformers import GPT2PreTrainedModel, GPT2Model
from transformers import top_k_top_p_filtering
from transformers.modeling_outputs import ModelOutput
from train.gpt2_model_train import model_train
import docx2txt
from utils.gpu_track import MemTracker
import inspect
from dataclasses import dataclass
from typing import Optional, Tuple

# device : GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
logging.basicConfig(level=logging.INFO)
# 追踪GPU Mem的消耗情况。
frame = inspect.currentframe()  # define a frame to track
gpu_tracker = MemTracker(frame)

# 训练参数
batch_size = 32
epochs = 5000
learning_rate = 1e-5  # 学习率
context_length = 512
action = 'train'  # train 训练   validate 测试  prod  生产运行
pretrained_model_dir = "../models/gpt2-chinese-cluecorpussmall/"
model_output_dir = "../models/chatgpt-aia-chinese/gpt-aia-chinese"


# 但这样有时可能会出现问题，例如模型陷入一个循环，不断生成同一个单词。
# 为了避免这种情况， GPT-2 设置了一个 top-k 参数，这样模型就会从概率前 k 大的单词中随机选取一个单词，作为下一个单词。
def select_top_k(predictions, k=10):
    predicted_tokens = random.choice(
        predictions[0, -1, :].sort(descending=True)[1][:10]).item()
    return predicted_tokens


'''
GPT-2 核心思想
(translate to french, english text, french text)
(answer the question, document, question, answer)
model类是目前在库中提供的8个模型架构的PyTorch模型(torch.nn.Modules)，例如BertModel
configuration类，它存储构建模型所需的所有参数，例如BertConfig。您不必总是自己实例化这些配置，特别是如果您使用的是未经任何修改的预训练的模型，创建模型将自动负责实例化配置(它是模型的一部分)
tokenizer类，它存储每个模型的词汇表，并在要输送到模型的词汇嵌入索引列表中提供用于编码/解码字符串的方法，例如BertTokenizer
from_pretraining()允许您从一个预训练版本实例化一个模型/配置/tokenizer
save_pretraining()允许您在本地保存模型/配置/tokenizer
'''


def train():
    # gpu_tracker.track()
    # 初始化预训练模型
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir)
    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_dir,
        vocab_size=len(tokenizer),
        n_ctx=context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = GPT2LMHeadModel(config)
    model.to(device)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"model size: {model_size / 1000 ** 2:.1f}M parameters")
    # gpu_tracker.track()

    # 数据集
    # 通过新的文本集训练进行微调
    # doc_path = './datasets/document/'
    # files = os.listdir(doc_path)
    # doc_texts = []
    # for file in files:
    #     # f = open(doc_path + file, 'r', encoding='UTF-8')
    #     text = docx2txt.process(doc_path + file)
    #     doc_texts.append(text.replace("\n\n", "\n"))
    # for text in doc_texts:
    #     for i in range(len(text) // max_len):
    #         # 将字符串分段成长度为max_len为单位
    #         # dataset.append(tokenizer.encode(text=text[i * max_len:(i + 1) * max_len]))
    #         text = preprocess(text)
    #         encoding = tokenizer(text=text[i * max_len:(i + 1) * max_len],
    #                              truncation=True,
    #                              padding=True,
    #                              max_length=max_len,
    #                              return_tensors="pt").to(device)
    #         dataset.extend(encoding['input_ids'])
    # del doc_texts
    file = '../datasets/company_datasets/doc_dataset.txt'
    lines = io.open(file, encoding='UTF-8').read().strip().split('\n')
    texts = [l for l in lines]
    max_len = max([len(text) for text in texts])
    train_set = GeneDataset(tokenizer=tokenizer,
                            texts=texts,
                            length=max_len
                            )
    print('dataset''s shape = {0}'.format(train_set.shape))

    # 开始模型训练
    pre = time.time()
    model.train()

    model_train(
        tokenizer=tokenizer,
        model=model,
        dataset=train_set,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        model_dir=model_output_dir,
    )
    # trainer.evaluate()
    print('训练时间：', time.time() - pre)


# 模型测试
def infer_answer(input_text):
    # 加载预训练模型：
    tokenizer = AutoTokenizer.from_pretrained(model_output_dir)
    model = GPT2LMHeadModel.from_pretrained(model_output_dir,
                                            bos_token_id=tokenizer.bos_token_id,
                                            eos_token_id=tokenizer.eos_token_id,
                                            pad_token_id=tokenizer.eos_token_id)
    model.to(device)

    def answer(text, sample=True, top_p=1, temperature=0.7):
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

    total_predicted_text = answer(input_text)

    return "".join(total_predicted_text.split())


@dataclass
class CausalLMOutputWithCrossAttentions(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    value: Optional[torch.FloatTensor] = None


# Cell
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


# Cell
class GPT2HeadWithValueModel(GPT2PreTrainedModel):
    """The GPT2HeadWithValueModel class implements a GPT2 language model with a secondary, scalar head."""

    def __init__(self, config):
        super().__init__(config)
        config.num_labels = 1
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # 为了适应强化这个框架，我们还需要对原来的 GPT 模型进行一定的封装，其实主要就是加一层 value head（线性层），
        # 让它预测每一个 token 的价值（理解为值函数，将 token 的隐藏状态转化为一个标量值）。
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


# Cell
def respond_to_batch(model, queries, txt_len=20, top_k=0, top_p=1.0):
    """Sample text from language model."""
    input_ids = queries
    for i in range(txt_len):
        # Get Logits
        outputs = model(input_ids)
        next_token_logits = outputs[0][:, -1, :]
        next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        # Sample
        probs = softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
    return input_ids[:, -txt_len:]


if __name__ == '__main__':
    if action == 'train':
        train()
    elif action == 'validate':
        cont = True
        while cont:
            input_text = str(input("请输入/Please input： "))

            if input_text == "exit":
                cont = False
            else:
                output_text = infer_answer(input_text)
                print(output_text)
