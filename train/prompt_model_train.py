# coding=utf-8
# 基于prompt模型的T5 model train
"""
# Project Name: MyGPT
# File Name: prompt_model_train
# Author: NSNP577
# Creation at 2023/2/28 10:10
# IDE: PyCharm
# Describe: 
"""

import os, json, logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os, time
# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
from utils.dataset_util import PromptSrcTgtDataset
from utils.gpu_track import print_gpu_info
from random import sample
from torch_optimizer import Adafactor
from tqdm import tqdm

# rich: for a better display on terminal
from rich.table import Column, Table
from rich import box
from rich.console import Console

# device : GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
if device == 'gpu': print_gpu_info()  # 输出当前服务器的GPU信息。
logging.basicConfig(level=logging.INFO)
# 做一些相关的配置(打印显示；GPU设置)
# define a rich console logger
console = Console(record=True)

# 定义模型的参数 let's define model parameters specific to T5 (Text-To-Text Transfer Transformer)
model_params = {
    "MODEL": "./models/ClueAI/PromptCLUE-base",  # model_type
    "TRAIN_BATCH_SIZE": 8,  # training batch size, 8
    "VALID_BATCH_SIZE": 8,  # validation batch size,8
    "TRAIN_EPOCHS": 10,  # number of training epochs
    "VAL_EPOCHS": 10,  # number of validation epochs
    "LEARNING_RATE": 1e-4,  # learning rate
    "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text, 512
    "MAX_TARGET_TEXT_LENGTH": 64,  # max length of target text,64
    "SEED": 42,  # set seed for reproducibility
}


# 数据准备：将json文件转化为csv形式的文件。
def convert_json_to_csv(source_file, target_file):
    """将json文件转化为csv形式的文件。
       source_file:输入文件；
       target_file：转化后的文件
    """
    lines = open(source_file, 'r', encoding='utf8').readlines()
    print("length of lines:", len(lines))
    input_list = []
    output_list = []
    answer_choices_list = []
    type_list = []
    for i, line in enumerate(lines):
        # {"input": "以下内容为真：“滁县地区专员张友道说:大都架到高处了”那么下面的陈述：“张友道对身边的官员说了话。”是真的,假的,或未知？\n答案：", "target": "未知", "answer_choices": ["真的", "假的", "未知"], "type": "nli"}
        # 1)获得字段值
        json_string = json.loads(line.strip())
        input_ = json_string["input"].replace("\n", "_")
        output_ = json_string["target"]
        answer_choices_ = json_string.get("answer_choices", [])
        type_ = json_string["type"]
        if i < 10: print(i, "input:", input_, ";output:", output_)
        # 2)添加到列表中
        input_list.append(input_)
        output_list.append(output_)
        answer_choices_list.append(answer_choices_)
        type_list.append(type_)

    # 3)写成pandas的dataframe，以csv进行保存
    df = pd.DataFrame({'input': input_list,
                       'target': output_list,
                       'answer_choices': answer_choices_list,
                       'type': type_list,
                       })
    df.to_csv(target_file, index=False)


# to display dataframe in ASCII format
def display_df(df):
    """display dataframe in ASCII format"""

    table = Table(
        Column("source_text", justify="center"),
        Column("target_text", justify="center"),
        title="Sample Data",
        pad_edge=False,
        box=box.ASCII,
    )

    for i, row in enumerate(df.values.tolist()):
        table.add_row(row[0], row[1])


# training logger to log training progress
training_logger = Table(
    Column("Epoch", justify="center"),
    Column("Steps", justify="center"),
    Column("Loss", justify="center"),
    title="Training Status",
    pad_edge=False,
    box=box.ASCII,
)


def train(epoch, tokenizer, model, device, loader, optimizer):
    """
    用于训练的方法
    Function to be called for training with the parameters passed from main function

    """
    model.train()
    time1 = time.time()
    for _, data in tqdm(enumerate(loader, 0)):
        y = data["target_ids"].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()  # target, from start to end(except end of token, <EOS>). e.g. "你好吗？"
        lm_labels = y[:, 1:].clone().detach()  # target, for second to end.e.g."好吗？<EOS>"
        lm_labels[y[:,
                  1:] == tokenizer.pad_token_id] = -100  # releted to pad_token and loss. for detail, check here: https://github.com/Shivanandroy/T5-Finetuning-PyTorch/issues/3
        ids = data["source_ids"].to(device, dtype=torch.long)  # input. e.g. "how are you?"
        mask = data["source_mask"].to(device, dtype=torch.long)

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=y_ids,
            labels=lm_labels,
        )
        loss = outputs[0]
        # 每100步打印日志
        if _ % 100 == 0 and _ != 0:
            time2 = time.time()
            print(_, "epoch:" + str(epoch) + "-loss:" + str(loss.item()) + ";each step's time spent:" + str(
                float(time2 - time1) / float(_ + 0.0001)))
            # training_logger.add_row(str(epoch), str(_), str(loss))
            # console.print(training_logger)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate(epoch, tokenizer, model, device, loader, max_length):
    """
    用于验证的方法：输入用于验证的数据，返回模型预测的结果和正确的标签
    Function to evaluate model for predictions

    """
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype=torch.long)
            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=max_length,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
            )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                     generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]
            if _ % 1000 == 0:
                # console.print(f'Completed {_}')
                logging.info('complete')

            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals


# 训练类：整合数据集类、训练方法、验证方法，加载数据进行训练并验证训练过程的效果
def T5Trainer(
        dataframe, source_text, target_text, model_params, output_dir="./outputs/"
):
    """
    T5 trainer
    """
    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(model_params["SEED"])  # pytorch random seed
    np.random.seed(model_params["SEED"])  # numpy random seed
    torch.backends.cudnn.deterministic = True  # 设定返回的卷积算法将是确定的

    # logging
    logging.debug('loading model: {0}'.format(model_params["MODEL"]))

    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])

    # Defining the model. We are using PromptCLUE model and added a Language model layer on top for generation of prediction.
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
    model = model.to(device)

    # logging
    logging.debug(f"[Data]: Reading data...\n")

    # Importing the raw dataset
    dataframe = dataframe[[source_text, target_text]]
    # display_df(dataframe.head(2))

    # Creation of Dataset and Dataloader
    # Defining the train size So 94% of the data will be used for training and the rest for validation.
    train_size = 0.94
    train_dataset = dataframe.sample(frac=train_size, random_state=model_params["SEED"])
    val_dataset = dataframe.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    # 打印数据集相关日志：数据量、训练步数
    logging.info(f"FULL Dataset: {dataframe.shape}")
    logging.info(f"TRAIN Dataset: {train_dataset.shape}")
    logging.info(f"TEST Dataset: {val_dataset.shape}\n")
    total_train_steps = int((train_dataset.shape[0] * model_params["TRAIN_EPOCHS"]) / model_params["TRAIN_BATCH_SIZE"])
    logging.info(f"Total Train Steps: {total_train_steps}\n")

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = PromptSrcTgtDataset(
        train_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )
    val_set = PromptSrcTgtDataset(
        val_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )

    # Defining the parameters for creation of dataloaders
    train_params = {
        "batch_size": model_params["TRAIN_BATCH_SIZE"],
        "shuffle": True,
        "num_workers": 0,
    }

    val_params = {
        "batch_size": model_params["VALID_BATCH_SIZE"],
        "shuffle": False,
        "num_workers": 0,
    }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    # optimizer = torch.optim.Adam(
    #     params=model.parameters(), lr=model_params["LEARNING_RATE"]
    # )
    optimizer = Adafactor(
        model.parameters(),
        lr=model_params["LEARNING_RATE"],
        eps2=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False
    )

    # Training loop
    logging.info(f"[Initiating Fine Tuning]...\n")
    logging.info('total epoch: {0}'.format(model_params["TRAIN_EPOCHS"]))
    for epoch in range(model_params["TRAIN_EPOCHS"]):
        print(f"epoch = {epoch}")
        # 1) train for one epoch
        train(epoch, tokenizer, model, device, training_loader, optimizer)

        # 2) save model for each epoch
        logging.info(f"[Saving Model] epoch = {epoch}..\n")
        path = os.path.join(output_dir, "model_files")
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)

        # 3) evaluating test dataset
        logging.info(f"[Initiating Validation]...\n")
        with torch.no_grad():  # add 2022.10.4
            # for epoch in range(model_params["VAL_EPOCHS"]):
            predictions, actuals = validate(epoch, tokenizer, model, device, val_loader,
                                            model_params["MAX_TARGET_TEXT_LENGTH"])
            final_df = pd.DataFrame({"Generated Text": predictions, "Actual Text": actuals})
            final_df.to_csv(os.path.join(output_dir, "predictions.csv"))

    logging.debug(os.path.join(output_dir, "logs.txt"))

    logging.info(f"[Validation Completed.]\n")
    logging.info(
        f"""[Model] Model saved @ {os.path.join(output_dir, "model_files")}\n"""
    )
    logging.info(
        f"""[Validation] Generation on Validation data saved @ {os.path.join(output_dir, 'predictions.csv')}\n"""
    )
    logging.info(f"""[Logs] Logs saved @ {os.path.join(output_dir, 'logs.txt')}\n""")


if __name__ == '__main__':
    # 处理数据集
    # 请运行以下三行代码进行格式换行，如果你需要全量数据训练。
    # 默认将只使用部分在线的示例数据进行训练。
    source_file = '../datasets/prompt/pCLUE_train.json'
    target_file = '../datasets/prompt/pCLUE_train.csv'
    convert_json_to_csv(source_file, target_file)

    # 训练模型
    # 使用 pCLUE:1200000+多任务提示学习数据集 的部分数据
    # dataframe必须有2列:
    #   - input: 文本输入
    #   - target: 目标输出
    df = pd.read_csv('../datasets/prompt/pCLUE_train.csv')  # 数据量：1200k数据。
    df = df.sample(frac=0.01)  # 测试只取1%作为训练样本
    print("df.head:", df.head(n=5))
    print("df.shape:", df.shape)
    # 显存占用说明：如果运行现在显存不足，请使用nvidia-smi查看显存；如果显卡多数被占用了，请重启colab程序
    T5Trainer(
        dataframe=df,
        source_text="input",
        target_text="target",
        model_params=model_params,
        output_dir="./models/CompanyModel0.1-Prompt-Chinese",
    )
