import torch
from torch import nn
import pytorch_lightning as pl
import torchmetrics
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import numpy as np

labels = ['background',
 'brighter',
 'color',
 'darker',
 'larger',
 'multiple_objects',
 'object_blocking',
 'partial_view',
 'pattern',
 'person_blocking',
 'pose',
 'shape',
 'smaller',
 'style',
 'subcategory',
 'texture']



# Define the ResNet-50 model
class ResNet(pl.LightningModule):
    def __init__(self, args, num_classes=16):
        super().__init__()
        self.resnet50 = torch.hub.load(
            "pytorch/vision:v0.9.0",
            "resnet50",
            weights="ResNet50_Weights.IMAGENET1K_V1",
        )
        self.resnet50.fc = nn.Linear(int(2048), int(num_classes))
        self.accuracy1 = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.accuracy3 = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=3)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
        self.confusion_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes, normalize="true")
        self.args = args

    def forward(self, x):
        return self.resnet50(x)

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
        
        self.confusion_matrix.update(preds=y_hat, target=y)
		

    def on_validation_epoch_end(self):
        confmat = self.confusion_matrix.compute().cpu() # .numpy()

        #log to wandb
        f, ax = plt.subplots(figsize = (15,10)) 
        sns.heatmap(confmat, ax=ax, annot=True, xticklabels=labels, yticklabels=labels)
        ax.set_xlabel('Predicted labels',size=15)
        ax.set_ylabel('True labels', size=15)
        ax.set_title(f'Confusion Matrix with sum {torch.sum(confmat)}', size=15)
        self.logger.experiment.log({"plot": wandb.Image(f) })
        
        self.confusion_matrix.reset()

        
    def configure_optimizers(self):
        # Choose optimizer from dict
        dictOptimizer = {
            "adam": torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay),
             "sgd": torch.optim.SGD(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)}
        optimizer = dictOptimizer[self.args.optimizer]
        #optimizer = torch.optim.Adam(
        #    self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        #)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, Tmax=self.args.epochs) # StepLR -> cosine
        return [optimizer], [scheduler]
    