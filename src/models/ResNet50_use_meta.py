import torch
from torch import nn
import pytorch_lightning as pl
import torchmetrics
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import numpy as np


# Define the ResNet-50 model
class ResNet_withMeta(pl.LightningModule):
    def __init__(self, num_classes=1000):
        super().__init__()
        #self.resnet50 = torch.hub.load(
        #    "pytorch/vision:v0.9.0",
        #    "resnet50",
        #    weights="ResNet50_Weights.IMAGENET1K_V1",
        #)
        #self.resnet50.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(int(2048), int(num_classes)))


        self.img_backbone = torch.hub.load(
            "pytorch/vision:v0.9.0",
            "resnet50",
            weights="ResNet50_Weights.IMAGENET1K_V1",
        )
        self.img_backbone.fc = nn.Identity() # nn.Sequential(nn.Dropout(0.5), nn.Linear(int(2048), int(1000)))
        
        self.meta_backbone = nn.Linear(16, 16)

        self.classifier = nn.Linear(2048 + 16, num_classes)
        
        # TODO change model here to use meta data (size 16) with size 1000 just before fully connected layer


        self.accuracy1 = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.accuracy3 = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, top_k=3
        )
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="micro")

        self.register_buffer
        

    def forward(self, img, meta):

        features_img = self.img_backbone(img)  # [N, 2048]
        features_meta = self.meta_backbone(meta)  # [N, n_features]
        features = torch.cat((features_img, features_meta), 0)

        return self.classifier(features)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)


        self.log("val_loss", loss)
        self.log("val_acc1", self.accuracy1(y_hat, y))
        self.log("val_acc3", self.accuracy3(y_hat, y))
        self.log("val_f1", self.f1_score(y_hat, y))


    def on_validation_epoch_end(self):
        pass

    def configure_optimizers(self):
        # Choose optimizer from dict
        dictOptimizer = {
            "adam": torch.optim.Adam(
                self.parameters(), lr=0.0001, weight_decay=0.0005
            ),
            "sgd": torch.optim.SGD(
                self.parameters(),
                lr=0.0001,
                weight_decay=0.0005,
                momentum=0.9,
            ),
        }
        optimizer = torch.optim.SGD(
                self.parameters(),
                lr=0.0001,
                weight_decay=0.0005,
                momentum=0.9 )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50
        )  # StepLR -> cosine
        return [optimizer], [scheduler]


