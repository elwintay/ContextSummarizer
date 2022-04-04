from operator import mul
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup, pipeline
import pytorch_lightning as pl
from torchmetrics.functional import f1
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from dataloader import *
from sklearn.metrics import classification_report

class BertTransformer(pl.LightningModule):

    def __init__(self, model_name="bert-base-uncased", n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        self.model = BertModel.from_pretrained(model_name, return_dict=True)
        self.classifier = nn.Linear(self.model.config.hidden_size, 6)
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion = nn.BCELoss()

    def forward(self, input_ids, attention_mask, labels=None):
        input = self.model(input_ids, attention_mask=attention_mask)
        logits = self.classifier(input.pooler_output)
        output = torch.sigmoid(logits) 
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
        return {"loss": loss, "predictions": outputs, "labels": labels}

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
        label_columns = ["Where is the location?","Who was the attacker?","Which organisation?","What was targeted?","Who injured or killed?","What weapon was used?"]

        for i, name in enumerate(label_columns):
            class_f1 = f1(predictions[:, i], labels[:, i], num_classes=2, multiclass=True)
            self.logger.experiment.add_scalar(f"{name}_f1/Train", class_f1, self.current_epoch)

    def validation_epoch_end(self, outputs):

        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)
        label_columns = ["Where is the location?","Who was the attacker?","Which organisation?","What was targeted?","Who injured or killed?","What weapon was used?"]

        for i, name in enumerate(label_columns):
            class_f1 = f1(predictions[:, i], labels[:, i], num_classes=2, multiclass=True)
            self.logger.experiment.add_scalar(f"{name}_f1/Valid", class_f1, self.current_epoch)

    def test_epoch_end(self, outputs):

        print(outputs)
        labels = []
        predictions = []
        docids = []
        texts = []
        for output in outputs:
            for out_labels, out_predictions, docid, text in zip(output["labels"],output["predictions"],output["docid"],output["text"]):
                labels.append(out_labels)
                predictions.append(out_predictions)
                docids.append(docid)
                texts.append(text)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)
        print(predictions)
        label_columns = ["Where is the location?","Who was the attacker?","Which organisation?","What was targeted?","Who injured or killed?","What weapon was used?"]

        for i, name in enumerate(label_columns):
            class_f1 = f1(predictions[:, i], labels[:, i], num_classes=2, multiclass=True)
            self.logger.experiment.add_scalar(f"{name}_f1/Valid", class_f1, self.current_epoch)

        results = {"preds":predictions, "labels":labels, "docid":docids, "text":texts}
        self.test_results = results

        return results


    def configure_optimizers(self):

        optimizer = AdamW(self.parameters(), lr=2e-5, no_deprecation_warning=True)

        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=self.n_warmup_steps,
                                                    num_training_steps=self.n_training_steps)

        return dict(optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval='step'))