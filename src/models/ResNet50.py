import torch
from torch import nn
import pytorch_lightning as pl
import torchmetrics
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import numpy as np

labels = {
    4: ["background", "color", "pattern", "pose"],
    16: [
        "background",
        "brighter",
        "color",
        "darker",
        "larger",
        "multiple_objects",
        "object_blocking",
        "partial_view",
        "pattern",
        "person_blocking",
        "pose",
        "shape",
        "smaller",
        "style",
        "subcategory",
        "texture",
    ],
}


# Define the ResNet-50 model
class ResNet(pl.LightningModule):
    def __init__(self, args, normed_weights=[], num_classes=16):
        super().__init__()
        self.resnet50 = torch.hub.load(
            "pytorch/vision:v0.9.0",
            "resnet50",
            weights="ResNet50_Weights.IMAGENET1K_V1",
        )
        # TODO Add dropout
        #self.resnet50.fc = nn.Linear(int(2048), int(num_classes))
        # self.resnet50.fc = nn.Linear(int(2048), int(16))
        self.resnet50.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(int(2048), int(num_classes)))
        
        self.accuracy1 = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.accuracy3 = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, top_k=3
        )
        self.f1_score_micro = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="micro"
        )
        self.f1_score_macro = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.confusion_matrix = torchmetrics.ConfusionMatrix(
            task="multiclass", num_classes=num_classes, normalize="true"
        )
        self.args = args
        self.register_buffer
        if self.args.weighted_loss:
            self.normed_weights = normed_weights.to("cuda")

        self.labels = labels[num_classes]

    def forward(self, x):
        return self.resnet50(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if self.args.weighted_loss:
            loss = nn.CrossEntropyLoss(weight=self.normed_weights)(y_hat, y)
        else:
            loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if self.args.weighted_loss:
            loss = nn.CrossEntropyLoss(weight=self.normed_weights)(y_hat, y)
        else:
            loss = nn.CrossEntropyLoss()(y_hat, y)

        self.log("val_loss", loss)
        self.log("val_acc1", self.accuracy1(y_hat, y))
        self.log("val_acc3", self.accuracy3(y_hat, y))
        self.log("val_f1_macro", self.f1_score_macro(y_hat, y))
        self.log("val_f1_micro", self.f1_score_micro(y_hat, y))

        self.confusion_matrix.update(preds=y_hat, target=y)

    def on_validation_epoch_end(self):
        confmat = self.confusion_matrix.compute().cpu()  # .numpy()

        # log to wandb
        f, ax = plt.subplots(figsize=(15, 10))
        sns.heatmap(
            confmat, ax=ax, annot=True, xticklabels=self.labels, yticklabels=self.labels
        )
        ax.set_xlabel("Predicted labels", size=15)
        ax.set_ylabel("True labels", size=15)
        ax.set_title(f"Confusion Matrix with sum {torch.sum(confmat)}", size=15)
        self.logger.experiment.log({"plot": wandb.Image(f)})

        self.confusion_matrix.reset()

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
