import torch
from torch import nn
import pytorch_lightning as pl
import torchmetrics
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import numpy as np


# Define the ResNet-50 model
class ResNetMultiLabel(pl.LightningModule):
    def __init__(self, args, normed_weights=[], num_labels=16):
        super().__init__()
        self.resnet50 = torch.hub.load(
            "pytorch/vision:v0.9.0",
            "resnet50",
            weights="ResNet50_Weights.IMAGENET1K_V1",
        )
        # TODO Fix final layer
        self.resnet50.fc = nn.Linear(
            in_features=self.resnet50.fc.in_features, out_features=num_labels
        )
        self.sigm = nn.Sigmoid()

        # self.resnet50.fc = nn.Linear(int(2048), int(16))
        self.accuracy1 = torchmetrics.Accuracy(task="multilabel", num_labels=num_labels)
        self.accuracy3 = torchmetrics.Accuracy(
            task="multilabel", num_labels=num_labels, top_k=3
        )
        self.f1_score = torchmetrics.F1Score(task="multilabel", num_labels=num_labels)
        self.threshold = 0.5
        self.args = args
        self.register_buffer
        print(f"Device is {self.device}")

    def forward(self, x):
        return self.sigm(self.resnet50(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.BCELoss()(y_hat, y.type(torch.float))
        self.log("train_loss", loss)

        y_hat_thresh = torch.where(y_hat > self.threshold, 1, 0)
        print(y)
        print()
        print(y_hat_thresh)
        print(self.accuracy1(y_hat_thresh, y))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.BCELoss()(y_hat, y.type(torch.float))

        y_hat_thresh = torch.where(y_hat > self.threshold, 1, 0)
        self.log("val_loss", loss)
        self.log("val_acc1", self.accuracy1(y_hat_thresh, y))
        self.log("val_acc3", self.accuracy3(y_hat_thresh, y))
        self.log("val_f1", self.f1_score(y_hat_thresh, y))

    def on_validation_epoch_end(self):
        pass

    def configure_optimizers(self):
        # Choose optimizer from dict
        dictOptimizer = {
            "adam": torch.optim.Adam(
                self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
            ),
            "sgd": torch.optim.SGD(
                self.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
                momentum=self.args.momentum,
            ),
        }
        optimizer = dictOptimizer[self.args.optimizer]
        # optimizer = torch.optim.Adam(
        #    self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        # )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.args.epochs
        )  # StepLR -> cosine
        return [optimizer], [scheduler]
