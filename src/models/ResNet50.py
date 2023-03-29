import torch
from torch import nn
import pytorch_lightning as pl
import torchmetrics
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import numpy as np



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
        self.confusion_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes)
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
		#print(cm)
        #log to wandb
        #fig, ax = plt.subplots() 
        #sns.heatmap(self.confusion_matrix.compute().cpu().numpy(), annot=True, ax=ax)
        #plt.savefig('confusion matrix')
        #self.logger.experiment.add_figure("val_confmat", fig)
        #self.logger.experiment.log({"confusion_matrix": wandb.Image(cm)})

    def on_validation_epoch_end(self):
        confmat = self.confusion_matrix.compute().cpu() # .numpy()

        #log to wandb
        f, ax = plt.subplots(figsize = (15,10)) 
        sns.heatmap(confmat, annot=True, ax=ax, )
        ax.set_xlabel('Predicted labels',size=15)
        ax.set_ylabel('True labels', size=15)
        ax.set_title(f'Confusion Matrix with sum {np.sum(confmat)}', size=15)
        self.logger.experiment.log({"plot": wandb.Image(f) })
        
        self.confusion_matrix.reset()
        # log the confusion matrix as an image
    #    self.logger.experiment.log({"confusion_matrix": wandb.Image(confmat)})
        # reset the confusion matrix for the next epoch
    #    self.confusion_matrix.reset()

        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        return [optimizer], [scheduler]