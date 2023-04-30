import torch
from torch import nn
import pytorch_lightning as pl
import torchmetrics
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import numpy as np
import pdb


# Define the ResNet-50 model
class ResNet_withMeta(pl.LightningModule):
    def __init__(self, args, num_classes=1000):
        super().__init__()

        self.img_backbone = torch.hub.load(
            "pytorch/vision:v0.9.0",
            "resnet50",
            weights="ResNet50_Weights.IMAGENET1K_V1",
        )
        self.meta_backbone = nn.Linear(16, 16)

        self.classifier = nn.Linear(2048 + 16, num_classes)

        self.args = args

        self.accuracy1 = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.accuracy3 = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, top_k=3
        )
        self.f1_score = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="micro"
        )

    def forward(self, img, meta):
        features_img = self.img_backbone(img)  # [N, 2048]
        features_meta = self.meta_backbone(meta)  # [N, n_features]
        features = torch.cat((features_img, features_meta), 1)

        return self.classifier(features)

    def training_step(self, batch, batch_idx):
        img, meta, y = batch
        y_hat = self(img, meta)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log("train_loss", loss)
        
        return loss

    def validation_step(self, batch, batch_idx):
        img, meta, y = batch
        y_hat = self(img, meta)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log("val_loss", loss)
        self.log("val_acc1", self.accuracy1(y_hat, y))
        self.log("val_acc3", self.accuracy3(y_hat, y))
        self.log("val_f1", self.f1_score(y_hat, y))

    def on_validation_epoch_end(self):
        pass

    def configure_optimizers(self):
        # Choose optimizer from dict
        from itertools import chain

        dictOptimizer = {
            "adam": torch.optim.Adam(
                chain(
                    self.img_backbone.fc.parameters(),
                    self.classifier.parameters(),
                    self.meta_backbone.parameters(),
                ),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
            ),
            "sgd": torch.optim.SGD(
                chain(
                    self.img_backbone.fc.parameters(),
                    self.classifier.parameters(),
                    self.meta_backbone.parameters(),
                ),
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
