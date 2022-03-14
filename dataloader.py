import numpy as np
import pandas as pd
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
import pytorch_lightning as pl

class MucDataset(Dataset):

    def __init__(self, data, max_token_len: int = 512):
        self.data = data
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):

        docid = self.data.loc[index,['docid']].values[0]
        qns = self.data.loc[index,['question']].values[0]
        sent = self.data.loc[index,['sentence']].values[0]
        text = "{0}[SEP]{1}".format(qns,sent)
        label = self.data.loc[index,['label']].astype(int).values

        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt')

        return dict(docid=docid, text=text, input_ids=encoding["input_ids"].flatten(),
                    attention_mask=encoding["attention_mask"].flatten(),
                    labels=torch.FloatTensor(label))

class MucDataModule(pl.LightningDataModule):

    def __init__(self, train_df, dev_df, test_df, workers=12, batch_size=1, max_token_len=512):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.dev_df = dev_df
        self.test_df = test_df
        self.max_token_len = max_token_len
        self.workers = workers

    def setup(self, stage=None):
        
        self.train_dataset = MucDataset(self.train_df,
                                        self.max_token_len)
        
        self.dev_dataset = MucDataset(self.dev_df,
                                      self.max_token_len)

        self.test_dataset = MucDataset(self.test_df,
                                       self.max_token_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.workers)

    def val_dataloader(self):
        return DataLoader(self.dev_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.workers)