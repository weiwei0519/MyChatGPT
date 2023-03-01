# coding=UTF-8
# my dataset definition

'''
@File: InputOutputDataset
@Author: WeiWei
@Time: 2023/2/25
@Email: weiwei_519@outlook.com
@Software: PyCharm
'''

import torch
from torch.utils.data import Dataset
from random import sample
from datasets import load_dataset


def preprocess(text):
    text = text.replace("\n", "\\n").replace("\t", "\\t")
    return text


def postprocess(text):
    return text.replace("\\n", "\n").replace("\\t", "\t")


# Input text to Output text dataset
class InputOutputDataset(Dataset):
    """
    创建一个自定义的数据集，用于训练，必须包括两个字段：输入(如source_text)、输出（如target_text）
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model
    """

    def __init__(
            self, tokenizer, source_len, target_len, source_texts, target_texts
    ):
        """
        Initializes a Dataset class

        Args:
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_texts (str): column name of source text
            target_texts (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.source_len = source_len
        self.target_len = target_len
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.source_shape = (len(source_texts), source_len)
        self.target_shape = (len(target_texts), target_len)

    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_texts)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        source_text = preprocess(str(self.source_texts[index]))
        target_text = preprocess(str(self.target_texts[index]))

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.target_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": torch.squeeze(source.data['input_ids']),
            "attention_mask": torch.squeeze(source.data['attention_mask']),
            "labels": torch.squeeze(target.data['input_ids'])
        }

    def val_sample(self, sample_rate=0.1):
        indexs = sample(list(range(self.source_shape[0])), int(sample_rate * self.source_shape[0]))
        val_dataset = InputOutputDataset(tokenizer=self.tokenizer,
                                         source_len=self.source_len,
                                         target_len=self.target_len,
                                         source_texts=[self.source_texts[i] for i in indexs],
                                         target_texts=[self.target_texts[i] for i in indexs],
                                         )
        return val_dataset


# Generative Dataset 生成式数据集
class GeneDataset(Dataset):
    """
    创建一个自定义的数据集，用于文本生成
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model
    """

    def __init__(
            self, tokenizer, length, texts
    ):
        """
        Initializes a Dataset class

        Args:
            tokenizer (transformers.tokenizer): Transformers tokenizer
            length (int): Max length of text
            texts：test list
        """
        self.tokenizer = tokenizer
        self.max_len = length
        self.texts = texts
        self.shape = (len(texts), self.max_len)

    def __len__(self):
        """returns the length of dataframe"""

        return len(self.texts)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        text = preprocess(str(self.texts[index]))

        # cleaning data so as to ensure data is in string type
        text = " ".join(text.split())

        text_encoder = self.tokenizer.batch_encode_plus(
            [text],
            max_length=self.max_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        text_encoder.data['labels'] = text_encoder.data['input_ids']  # 文本生成的自回归任务。

        return text_encoder.data

    def val_sample(self, sample_rate=0.1):
        indexs = sample(list(range(self.shape[0])), int(sample_rate * self.shape[0]))
        val_dataset = GeneDataset(tokenizer=self.tokenizer,
                                  length=self.max_len,
                                  texts=[self.texts[i] for i in indexs]
                                  )
        return val_dataset


class TextClassifyDataset(Dataset):
    def __init__(self, tokenizer, texts, labels, max_len):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.shape = (len(texts), max_len)

    def __getitem__(self, index):
        text = preprocess(str(self.texts[index]))

        text_encode = self.tokenizer.batch_encode_plus(
            [text],
            max_length=self.shape[1],
            pad_to_max_length=True,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        item = {
            'input_ids': torch.squeeze(text_encode.data['input_ids']),
            # 'token_type_ids': torch.squeeze(text_encode.data['token_type_ids']),
            'attention_mask': torch.squeeze(text_encode.data['attention_mask']),
            'labels': torch.tensor(self.labels[index])
        }
        return item

    def __len__(self):
        return self.shape[0]


class PromptSrcTgtDataset(Dataset):
    """
    创建一个自定义的数据集，用于训练，必须包括两个字段：输入(如source_text)、输出（如target_text）
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
            self, dataframe, tokenizer, source_len, target_len, source_text, target_text
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = self.data[target_text]
        self.source_text = self.data[source_text]

    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_text)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }
