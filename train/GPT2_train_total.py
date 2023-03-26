# coding=UTF-8
# 

'''
@File: GPT2_train_total
@Author: WeiWei
@Time: 2023/3/23
@Email: weiwei_519@outlook.com
@Software: PyCharm
'''

# project external dependency
import os
import argparse
import time
import logging
import glob
import torch
from torch import nn
from torch.nn import Identity, CrossEntropyLoss
from torch import softmax
from torch.optim import AdamW
from torch.utils.data import DataLoader
import numpy as np
import docx2txt
import pandas as pd
from tqdm import tqdm
from rouge import Rouge
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers import get_scheduler
from transformers.models.gpt2 import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from transformers.utils import ModelOutput

# Project internal dependency
from utils.dataset_util import GPT2Dataset, preprocess, postprocess

# device : GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
logging.basicConfig(level=logging.INFO)
pad_token_id = 0


# 训练参数
def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0', type=str,
                        help='cuda:n——training with GPU, cpu——training with CPU')
    parser.add_argument('--pretrained_model_dir', default="../models/chatgpt-aia-chinese/gpt-aia-chinese/", type=str, help='')
    parser.add_argument('--model_output_dir', default="../models/chatgpt-aia-chinese/gpt-aia-chinese/", type=str,
                        help='')
    parser.add_argument('--doc_path', default='../datasets/company_datasets/aiacn/', type=str, help='')
    parser.add_argument('--action', default='validate', type=str,
                        help='train训练/validate测试/checkpoint继续训练/fine-tuning微调模型/prod生产运行')
    parser.add_argument('--with_pretrained_mode', default=False, type=str,
                        help='采用空模型，还是预训练模型')
    parser.add_argument('--batch_size', default=2, type=int, required=False, help='batch size')
    parser.add_argument('--epochs', default=10000, type=int, required=False, help='epochs')
    parser.add_argument('--max_length', default=200, type=int, required=False, help='max context length')
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up steps')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='learn rate')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--log_step', default=10, type=int, required=False, help='print log steps')
    return parser.parse_args()


def rouge(not_ignore, shift_labels, preds):
    main_rouge = Rouge()
    true_length = [w.sum() for w in not_ignore.float()]
    rouge_labels = []
    rouge_predicts = []
    for idx, tmp_len in enumerate(true_length):
        tmp_labels = shift_labels[idx][:int(tmp_len)]
        rouge_labels.append(" ".join([str(w) for w in tmp_labels.tolist()]))
        tmp_pred = preds[idx][:int(tmp_len)]
        rouge_predicts.append(" ".join([str(w) for w in tmp_pred.tolist()]))
    rouge_score = main_rouge.get_scores(rouge_predicts, rouge_labels, avg=True)
    return rouge_score


def calculate_loss_and_accuracy(outputs, labels, device):
    global pad_token_id
    logits = outputs.logits
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(device)

    # Flatten the tokens
    loss_fct = CrossEntropyLoss(ignore_index=pad_token_id, reduction='sum')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    _, preds = shift_logits.max(dim=-1)
    not_ignore = shift_labels.ne(pad_token_id)
    num_targets = not_ignore.long().sum().item()

    correct = (shift_labels == preds) & not_ignore
    correct = correct.float().sum()

    accuracy = correct / num_targets
    loss = loss / num_targets

    rouge_score = rouge(not_ignore, shift_labels, preds)
    return loss, accuracy, rouge_score


def collate_fn(batch):
    global pad_token_id
    input_ids = []
    input_lens_list = [len(w) for w in batch]
    max_input_len = max(input_lens_list)
    for btc_idx in range(len(batch)):
        input_len = len(batch[btc_idx])
        input_ids.append(batch[btc_idx])
        input_ids[btc_idx].extend([pad_token_id] * (max_input_len - input_len))
    # print(f"input_ids shape: {len(input_ids)}x{len(input_ids[0])}")
    # for l, input_id in enumerate(input_ids):
    #     for r, id in enumerate(input_id):
    #         if not isinstance(id, int):
    #             print(f"line:{l}, row:{r}, id:{id}")
    return torch.tensor(input_ids, dtype=torch.long)


def data_loader(args, doc_path, tokenizer, shuffle):

    # 初始训练，基于原始doc格式文本数据集进行GPT模型训练
    texts = []
    if args.action == 'train' or args.action == 'checkpoint':
        doc_path = '../datasets/company_datasets/aiacn/'
        files = os.listdir(doc_path)
        for file in files:
            doc_content = docx2txt.process(doc_path + file)
            # doc_content = "".join(doc_content.split())   # 去掉空格
            # 对于超出content_length限制的文本，需要进行拆分处理。
            if len(doc_content) <= args.max_length:
                texts.append(doc_content)
            else:
                r = 0
                for i in range(len(doc_content) // args.max_length):
                    texts.append(doc_content[i * args.max_length:(i + 1) * args.max_length])
                    r += 1
                texts.append(doc_content[r * args.max_length:len(doc_content)])
        input_ids = []
        for text in tqdm(texts):
            input_id = tokenizer.encode(text)
            input_ids.append(input_id)
    elif args.action == 'fine-tuning':
        doc_path = '../datasets/company_datasets/aiacn/Prompt_Finetuning.xlsx'
        content = pd.read_excel(doc_path)
        print(content.head(5))
        text_pairs = content.iloc[:, [0, 1]]
        source_texts = content.iloc[:, 0].values.tolist()
        target_texts = content.iloc[:, 1].values.tolist()
        paraphs = [src + tgt for src, tgt in zip(source_texts, target_texts)]

    dataset = GPT2Dataset(input_ids, tokenizer)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=shuffle,
                            collate_fn=collate_fn)

    return dataloader


def train(args, model, dataloader):
    # 开始模型训练
    pre = time.time()
    num_training_steps = args.epochs * len(dataloader)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps
    )
    model.train()
    batch_steps = 0
    for epoch in range(args.epochs):
        for batch in dataloader:
            batch_steps += 1
            inputs = {"input_ids": batch.to(device)}
            outputs = model(**inputs, labels=batch.to(device))
            # loss = outputs.loss
            loss, acc, rouge_score = calculate_loss_and_accuracy(outputs, batch.to(device), device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if batch_steps % args.log_step == 0:
                print("train epoch {}/{}, batch {}/{}, loss {}, accuracy {}, rouge-1 {}, rouge-2 {}, rouge-l {}".format(
                    epoch, args.epochs,
                    batch_steps,
                    num_training_steps,
                    loss, acc,
                    rouge_score["rouge-1"]['f'],
                    rouge_score["rouge-2"]["f"],
                    rouge_score["rouge-l"]["f"]))

        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(args.model_output_dir)
    # torch.save(model, os.path.join(args.final_model_path, 'gpt2_WenAn.pth'))

    print('训练时间：', time.time() - pre)


def evaluate(dataloader, args):
    model, _ = load_model(args)
    model.to(device)
    model.eval()
    loss_list, acc_list, rouge_1_list, rouge_2_list, rouge_l_list = [], [], [], [], []
    batch_steps = 0
    with torch.no_grad():
        for batch in dataloader:
            batch_steps += 1
            inputs = {"input_ids": batch.to(device)}
            outputs = model(**inputs, labels=batch.to(device))
            loss, acc, rouge_score = calculate_loss_and_accuracy(outputs, batch.to(device), device)
            loss_list.append(float(loss))
            acc_list.append(float(acc))
            rouge_1_list.append(float(rouge_score["rouge-1"]['f']))
            rouge_2_list.append(float(rouge_score["rouge-2"]['f']))
            rouge_l_list.append(float(rouge_score["rouge-l"]['f']))
            print("eval batch {}/{}, loss {}, accuracy {}, rouge-1 {}, rouge-2 {}, rouge-l {}".format(
                batch_steps,
                len(dataloader),
                loss, acc,
                rouge_score["rouge-1"]['f'],
                rouge_score["rouge-2"]["f"],
                rouge_score["rouge-l"]["f"]))
    print("loss: {},".format(np.mean(loss_list)),
          "accuracy: {}.".format(np.mean(acc_list)),
          "rouge-1: {},".format(np.mean(rouge_1_list)),
          "rouge-2: {},".format(np.mean(rouge_2_list)),
          "rouge-l: {}".format(np.mean(rouge_l_list)))


def load_model(args):
    # 加载预训练模型，若不存在则初始化空模型
    checkpoint = glob.glob(os.path.join(args.model_output_dir, 'checkpoint-*'))  # 按照目前trainer的训练输出，只会存在一个checkpoint
    if len(checkpoint) > 0 and args.action == 'checkpoint':
        # 从checkpoint断点继续训练
        checkpoint = (checkpoint[0]).replace("\\", "/")
        tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
        config = GPT2Config.from_pretrained(checkpoint)
        model = GPT2LMHeadModel.from_pretrained(checkpoint)
    elif args.action == 'fine-tuning':
        # 对GPT2模型进行调优
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_output_dir)
        config = GPT2Config.from_pretrained(args.model_output_dir)
        model = GPT2LMHeadModel.from_pretrained(args.model_output_dir)
    else:
        if args.with_pretrained_mode:
            # 加载预训练模型
            tokenizer = GPT2Tokenizer.from_pretrained(args.pretrained_model_dir)
            config = GPT2Config.from_pretrained(args.pretrained_model_dir)
            model = GPT2LMHeadModel.from_pretrained(args.pretrained_model_dir, config=config)
        else:
            # 加载空模型
            tokenizer = GPT2Tokenizer.from_pretrained(args.pretrained_model_dir)
            config = GPT2Config.from_pretrained(args.pretrained_model_dir)
            model = GPT2LMHeadModel(config=config)

    model.to(args.device)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"model size: {model_size / 1000 ** 2:.1f}M parameters")
    # gpu_tracker.track()
    return config, tokenizer, model


def model_train(args):
    # 加载预训练模型或空模型
    config, tokenizer, model = load_model(args=args)
    global pad_token_id
    pad_token_id = tokenizer.eos_token_id
    train_dataloader = data_loader(args, args.doc_path, tokenizer=tokenizer, shuffle=True)
    eval_dataloader = data_loader(args, args.doc_path, tokenizer=tokenizer, shuffle=False)

    print('dataset''s shape = {0}'.format(len(train_dataloader)))

    train(
        args=args,
        model=model,
        dataloader=train_dataloader
    )


@dataclass
class CausalLMOutputWithCrossAttentions(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    value: Optional[torch.FloatTensor] = None


# 模型测试
def infer_answer(input_text, tokenizer, model, do_sample=False, return_seqs=1):
    input_ids = []
    input_ids.extend(tokenizer.encode(input_text))
    input_ids = input_ids[:-1]
    for i in range(args.max_length):
        inputs = {"input_ids": torch.tensor([input_ids]).to(device)}
        outputs = model(**inputs)
        logits = outputs.logits
        last_token_id = int(torch.argmax(logits[0][-1]))
        last_token = tokenizer.convert_ids_to_tokens(last_token_id)
        input_text += tokenizer.convert_tokens_to_string(last_token)
        input_ids.append(last_token_id)
    return input_text


if __name__ == '__main__':
    args = setup_args()
    if args.action == 'train' or args.action == 'checkpoint' or args.action == 'fine-tuning':
        model_train(args)
    elif args.action == 'validate':
        # 加载预训练模型：
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_output_dir)
        model = GPT2LMHeadModel.from_pretrained(args.model_output_dir)
        model.to(device)
        cont = True
        while cont:
            sep = 100
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
                output_text = infer_answer(input_text=input_text, tokenizer=tokenizer, model=model)
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
