import numpy as np
import pandas as pd
import json
import torch
import ast
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, DistilBertTokenizerFast
import pytorch_lightning as pl

class MucDataset(Dataset):

    def __init__(self, data, tokenizer, max_token_len: int = 512):
        self.data = data
        self.max_token_len = max_token_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):

        docid = self.data.loc[index,['docid']].values[0]
        template = self.data.loc[index,['template']].values[0]
        sent = self.data.loc[index,['sentence']].values[0]
        text = "{0}[SEP]{1}".format(template,sent)
        label = self.data.loc[index,["Where is the location?","Who was the attacker?","Which organisation?","What was targeted?","Who injured or killed?","What weapon was used?"]].astype(int).values

        encoding = self.tokenizer.encode_plus(
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

    def __init__(self, train_df, dev_df, test_df, tokenizer, workers=12, batch_size=1, max_token_len=512):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.dev_df = dev_df
        self.test_df = test_df
        self.max_token_len = max_token_len
        self.workers = workers
        self.tokenizer = tokenizer

    def setup(self, stage=None):
        
        self.train_dataset = MucDataset(self.train_df,
                                        self.tokenizer,
                                        self.max_token_len)
        
        self.dev_dataset = MucDataset(self.dev_df,
                                      self.tokenizer,
                                      self.max_token_len)

        self.test_dataset = MucDataset(self.test_df,
                                       self.tokenizer,
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

class QADataset(Dataset):

    def __init__(self, data, tokenizer, max_token_len: int = 512):
        self.data = data[data['entity']!='{}'].reset_index(drop=True)
        self.max_token_len = max_token_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def add_token_positions(self, encodings, answers):
      
        # initialize lists to contain the token indices of answer start/end
        start_positions = []
        end_positions = []
        for i in range(len(answers)):
            # append start/end token position using char_to_token method
            start_positions.append(encodings.char_to_token(i, answers['start']))
            end_positions.append(encodings.char_to_token(i, answers['end']))

            # if start position is None, the answer passage has been truncated
            if start_positions[-1] is None:
                start_positions[-1] = tokenizer.model_max_length
            # end position cannot be found, char_to_token found space, so shift position until found
            shift = 1
            while end_positions[-1] is None:
                end_positions[-1] = encodings.char_to_token(i, answers['end'] - shift)
                shift += 1
        # update our encodings object with the new token-based start/end positions
        encodings.update({'start_positions': start_positions, 'end_positions': end_positions})
        print({'start_positions': start_positions, 'end_positions': end_positions})
        return encodings

    def __getitem__(self, index: int):

        docid = self.data.loc[index,['docid']].values[0]
        template = self.data.loc[index,['template']].values[0]
        qns = self.data.loc[index,['question']].values[0]
        sent = self.data.loc[index,['sentence']].values[0]
        text = "{0}[SEP]{1}[SEP]{2}".format(sent,template,qns)
        label = ast.literal_eval(self.data.loc[index,['entity']].values[0])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt')

        start_pos = encoding.char_to_token(label['start'])
        end_pos = encoding.char_to_token(label['end'])

        if start_pos is None:
            start_pos = tokenizer.model_max_length

        shift = 1
        while end_pos is None:
            end_pos = encoding.char_to_token(label['end'] - shift)
            shift += 1

        return dict(docid=docid, text=text, qns=qns, input_ids=encoding["input_ids"].flatten(),
                    attention_mask=encoding["attention_mask"].flatten(), 
                    start_positions=start_pos,
                    end_positions=end_pos)

class QADataModule(pl.LightningDataModule):

    def __init__(self, train_df, dev_df, test_df, tokenizer, workers=12, batch_size=1, max_token_len=512):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.dev_df = dev_df
        self.test_df = test_df
        self.max_token_len = max_token_len
        self.workers = workers
        self.tokenizer = tokenizer

    def setup(self, stage=None):
        
        self.train_dataset = QADataset(self.train_df,
                                       self.tokenizer,
                                       self.max_token_len)
        
        self.dev_dataset = QADataset(self.dev_df,
                                     self.tokenizer,
                                     self.max_token_len)

        self.test_dataset = QADataset(self.test_df,
                                      self.tokenizer,
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

if __name__ == "__main__":
    from transformers import DistilBertTokenizerFast
    test = pd.read_csv("{}/test_qa.csv".format("data/muc_sentence_6_fields"))
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    data = QADataset(test,tokenizer,512)
    print(data.__getitem__(200))