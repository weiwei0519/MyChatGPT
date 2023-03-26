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
import glob
import random
import time
from random import sample
from tqdm import tqdm
import io
from transformers import GPT2LMHeadModel, AutoTokenizer, AutoConfig, pipeline, set_seed, BertTokenizer
from transformers import GPT2PreTrainedModel, GPT2Model
from transformers import top_k_top_p_filtering
from transformers.modeling_outputs import ModelOutput
from train.gpt2_model_train import model_train
import docx2txt
import pandas as pd
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
batch_size = 2
epochs = 10000
learning_rate = 1e-5  # 学习率
text_length = 500
context_length = 1024
action = 'train'  # train 训练    validate 测试     prod 生产运行   checkpoint 继续训练     fine-tuning 微调模型
pretrained_model_dir = "../models/Wenzhong2.0-GPT2-3.5B-chinese/"
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
    global action
    # 加载aia预训练模型，若不存在则初始化空模型
    checkpoint = glob.glob(os.path.join(model_output_dir, 'checkpoint-*'))  # 按照目前trainer的训练输出，只会存在一个checkpoint
    if len(checkpoint) > 0 and action == 'checkpoint':
        # 从checkpoint断点继续训练
        checkpoint = (checkpoint[0]).replace("\\", "/")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        config = AutoConfig.from_pretrained(checkpoint)
        model = GPT2LMHeadModel.from_pretrained(checkpoint)
        model.to(device)
    elif action == 'fine-tuning':
        # 对GPT2模型进行调优
        tokenizer = AutoTokenizer.from_pretrained(model_output_dir)
        config = AutoConfig.from_pretrained(model_output_dir)
        model = GPT2LMHeadModel.from_pretrained(model_output_dir)
    else:
        # 初始化空模型
        tokenizer = AutoTokenizer.from_pretrained(model_output_dir)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=model_output_dir,
        )
        # the pre-trained model is loaded with the custom configuration
        model = GPT2LMHeadModel(config)
        # model = GPT2LMHeadModel.from_pretrained(model_output_dir, config=config)
        # num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        # the model embedding is resized
        # model.resize_token_embeddings(len(tokenizer))
        model.to(device)

    model_size = sum(t.numel() for t in model.parameters())
    print(f"model size: {model_size / 1000 ** 2:.1f}M parameters")
    # gpu_tracker.track()

    # 初始训练，基于原始doc格式文本数据集进行GPT模型训练
    texts = []
    paraphs = []
    if action == 'train' or action == 'checkpoint':
        doc_path = '../datasets/company_datasets/aiacn/'
        files = os.listdir(doc_path)
        for file in files:
            # paraphs.extend(docx2txt.process(doc_path + file).replace("\n\n", "\n").strip().split('\n'))
            paraphs.append(docx2txt.process(doc_path + file))
    elif action == 'fine-tuning':
        doc_path = '../datasets/company_datasets/aiacn/Prompt_Finetuning.xlsx'
        content = pd.read_excel(doc_path)
        print(content.head(5))
        text_pairs = content.iloc[:, [0, 1]]
        source_texts = content.iloc[:, 0].values.tolist()
        target_texts = content.iloc[:, 1].values.tolist()
        paraphs = [src + tgt for src, tgt in zip(source_texts, target_texts)]

    # 对于超出content_length限制的文本，需要进行拆分处理。
    for text in paraphs:
        # text = "".join(text.split())  # 去掉空格
        if len(text) <= text_length:
            texts.append(text)
        else:
            r = 0
            for i in range(len(text) // text_length):
                texts.append(text[i * text_length:(i + 1) * text_length])
                r += 1
            texts.append(text[r * text_length:len(text)])
    # for text in paraphs:
    #     text = "".join(text.split())  # 去掉空格
    #     length = len(text) + 2  # 需要加上bos和eos的长度
    #     if length <= max_length:
    #         texts.append(cls + text + eos + pad * (max_length - length))
    #     else:
    #         r = 0
    #         for i in range(length // max_length):
    #             if i == 0:
    #                 texts.append(cls + text[0:max_length - 1])
    #             else:
    #                 texts.append(text[i * max_length - 1:(i + 1) * max_length - 1])
    #             r = i
    #         texts.append(text[(r + 1) * max_length - 1:-1] + eos + pad * (
    #                 max_length - 1 - len(text[(r + 1) * max_length - 1:-1])))
    del paraphs
    # 基于texts文本数据list，创建GPT2模型数据集
    train_set = GeneDataset(tokenizer=tokenizer,
                            texts=texts,
                            length=context_length
                            )
    eval_set = GeneDataset(tokenizer=tokenizer,
                           texts=sample(texts, int(0.1 * len(texts))),
                           length=context_length
                           )
    print('dataset''s shape = {0}'.format(train_set.shape))

    # 开始模型训练
    pre = time.time()
    model.train()

    model_train(
        tokenizer=tokenizer,
        model=model,
        train_dataset=train_set,
        eval_dataset=eval_set,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        action=action,
        model_dir=model_output_dir,
    )
    # trainer.evaluate()
    print('训练时间：', time.time() - pre)


# 模型测试
def infer_answer(input_text, tokenizer, model, do_sample, return_seqs=1):
    '''sample：是否抽样。生成任务，可以设置为True;
    top_p：0-1之间，生成的内容越多样'''
    top_p = 1  # 已知生成各个词的总概率是1（即默认是1.0）如果top_p小于1，则从高到低累加直到top_p，取这前N个词作为候选。
    temperature = 0.7  # 默认是1.0，温度越低（小于1），softmax输出的贫富差距越大；温度越高，softmax差距越小。
    text = preprocess(input_text)
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    generated_text_samples = model.generate(input_ids,
                                            max_length=text_length,
                                            num_beams=return_seqs,
                                            no_repeat_ngram_size=2,
                                            do_sample=do_sample,
                                            num_return_sequences=return_seqs,
                                            early_stopping=True,
                                            )
    predict_texts = []
    for i, beam in enumerate(generated_text_samples):
        output_text = tokenizer.decode(beam, skip_special_tokens=True)
        # 替换special token
        output_text = output_text[0].replace(cls, '').replace(eos, '').replace(pad, '')
        predict_text = postprocess(output_text)
        predict_texts.append("".join(predict_text[i].split()) for i in range(len(predict_text)))


def answer(text, sample=True, top_p=1, temperature=0.7):
    '''sample：是否抽样。生成任务，可以设置为True;
    top_p：0-1之间，生成的内容越多样'''
    text = preprocess(text)
    encoding = tokenizer(text=[text], truncation=True, padding=True, max_length=768, return_tensors="pt").to(device)
    if not sample:
        out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_new_tokens=512,
                             num_beams=1, length_penalty=0.6)
    else:
        out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_new_tokens=512,
                             do_sample=True, top_p=top_p, temperature=temperature, no_repeat_ngram_size=3)
    out_text = tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)
    return postprocess(out_text[0])


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
    if action == 'train' or action == 'checkpoint' or action == 'fine-tuning':
        train()
    elif action == 'validate':
        # 加载预训练模型：
        tokenizer = AutoTokenizer.from_pretrained(model_output_dir)
        model = GPT2LMHeadModel.from_pretrained(model_output_dir)
        model.to(device)
        # generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
        set_seed(42)
        cont = True
        while cont:
            sep = 80
            input_text = str(input("请输入/Please input： "))

            if input_text == "exit":
                cont = False
            else:
                # output_text = infer_answer(input_text,
                #                            tokenizer=tokenizer,
                #                            model=model,
                #                            do_sample=False,
                #                            return_seqs=1,
                #                            )
                output_text = answer(input_text)
                if isinstance(output_text, list):
                    for text in output_text:
                        idx = 0
                        for i in range(len(text) // sep):
                            print(text[i * sep:(i + 1) * sep])
                            idx = i + 1
                        print(text[idx * sep:-1])
                else:
                    idx = 0
                    for i in range(len(output_text) // sep):
                        print(output_text[i * sep:(i + 1) * sep])
                        idx = i + 1
                    print(output_text[idx * sep:-1])
                # print(output_text)
