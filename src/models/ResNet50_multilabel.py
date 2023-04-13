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

        self.resnet50.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(
            in_features=self.resnet50.fc.in_features, out_features=num_labels
        ))
        self.sigm = nn.Sigmoid()

        # self.resnet50.fc = nn.Linear(int(2048), int(16))
        self.accuracy1 = torchmetrics.Accuracy(task="multilabel", num_labels=num_labels)
        self.accuracy3 = torchmetrics.Accuracy(
            task="multilabel", num_labels=num_labels, top_k=3
        )
        self.f1_score = torchmetrics.F1Score(task="multilabel", num_labels=num_labels, average="micro")
        self.f1_multi = torchmetrics.F1Score(task = "multilabel", num_labels=num_labels, average=None)
        self.threshold = 0.5
        self.args = args
        self.register_buffer
        self.normed_weights = normed_weights.to("cuda")

    def forward(self, x):
        return self.sigm(self.resnet50(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.BCELoss(reduction="none")(y_hat, y.type(torch.float))
        loss = torch.mean(loss * self.normed_weights.unsqueeze(0))

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.BCELoss(reduction="none")(y_hat, y.type(torch.float))
        loss = torch.mean(loss * self.normed_weights.unsqueeze(0))

        y_hat_thresh = torch.where(y_hat > self.threshold, 1, 0)

        self.f1_multi.update(preds=y_hat_thresh, target=y)

        self.log("val_loss", loss)
        self.log("val_acc1", self.accuracy1(y_hat_thresh, y))
        self.log("val_acc3", self.accuracy3(y_hat_thresh, y))
        self.log("val_f1", self.f1_score(y_hat_thresh, y))

    def on_validation_epoch_end(self):
        # Get the F1 score for each class
        f1_scores_per_class = self.f1_multi.compute().cpu().tolist()

        # Create a bar plot of F1 scores
        fig, ax = plt.subplots()
        ax.bar(range(len(f1_scores_per_class)), f1_scores_per_class)
        ax.set_xlabel('Class')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Score for each Class')
        self.logger.experiment.log({"plot": wandb.Image(fig)})

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
