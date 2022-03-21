from operator import mul
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup
import pytorch_lightning as pl
from torchmetrics.functional import f1
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from dataloader import *
from sklearn.metrics import classification_report
from focal_loss.focal_loss import FocalLoss

class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()


class BertTransformer(pl.LightningModule):

    def __init__(self, model_name="bert-base-uncased", n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        self.model = BertModel.from_pretrained(model_name, return_dict=True)
        self.classifier = nn.Linear(self.model.config.hidden_size, 6)
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        # self.criterion = FocalLoss(alpha=0.75,gamma=5)
        # self.criterion = WeightedFocalLoss(alpha=.25,gamma=5)
        self.criterion = nn.BCELoss()

    def forward(self, input_ids, attention_mask, labels=None):
        input = self.model(input_ids, attention_mask=attention_mask)
        logits = self.classifier(input.pooler_output)
        output = torch.sigmoid(logits) 
        loss = 0
        if labels is not None:
            # loss = self.criterion(logits, labels) #Use when using WeightedFocalLoss
            loss = self.criterion(output, labels) #Use when using FocalLoss or BCELoss
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

        # labels = []
        # predictions = []
        # for output in outputs:
        #     for out_labels, out_predictions in zip(output["labels"],output["predictions"]):
        #         labels.append(out_labels)
        #         predictions.append(out_predictions)

        # labels = torch.stack(labels).int()
        # predictions = torch.stack(predictions)

        # class_f1 = f1(predictions, labels)
        # print("Train F1: {}".format(class_f1))

        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)
        label_columns = ['What was targeted','What was used','Where','Which','Who attack','Who injured or killed']

        for i, name in enumerate(label_columns):
            class_f1 = f1(predictions[:, i], labels[:, i], num_classes=2, multiclass=True)
            self.logger.experiment.add_scalar(f"{name}_f1/Train", class_f1, self.current_epoch)

    def validation_epoch_end(self, outputs):

        # labels = []
        # predictions = []
        # for output in outputs:
        #     for out_labels, out_predictions in zip(output["labels"],output["predictions"]):
        #         labels.append(out_labels)
        #         predictions.append(out_predictions)

        # labels = torch.stack(labels).int().squeeze().tolist()
        # predictions = torch.stack(predictions).int().squeeze().tolist()

        # # class_f1 = f1(predictions, labels)
        # # print("Valid F1: {}".format(class_f1))
        # print(classification_report(labels, predictions, zero_division=0))

        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)
        label_columns = ['What was targeted','What was used','Where','Which','Who attack','Who injured or killed']

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
        label_columns = ['What was targeted','What was used','Where','Which','Who attack','Who injured or killed']

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