import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup
import pytorch_lightning as pl
from torchmetrics.functional import f1
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from dataloader import *
from sklearn.metrics import classification_report

class BertTransformer(pl.LightningModule):

    def __init__(self, model_name, n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        self.model = BertModel.from_pretrained(model_name, return_dict=True)
        self.classifier = nn.Linear(self.model.config.hidden_size, 1)
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion = nn.BCELoss()

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        text = batch['text']
        docid = batch['docid']
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels, "docid": docid, "text": text}

    def training_epoch_end(self, outputs):

        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)

        for i, name in enumerate(self.label_columns):
            class_f1 = f1(predictions[:, i], labels[:, i], num_classes=2)
            self.logger.experiment.add_scalar(f"{name}_f1/Train", class_f1, self.current_epoch)


    def configure_optimizers(self):

        optimizer = AdamW(self.parameters(), lr=2e-5, no_deprecation_warning=True)

        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=self.n_warmup_steps,
                                                    num_training_steps=self.n_training_steps)

        return dict(optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval='step'))