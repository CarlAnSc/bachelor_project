import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from torchvision import transforms
import pytorch_lightning as pl
import wandb
import argparse

parser = argparse.ArgumentParser(description='Training of model for hpc and local')
parser.add_argument('path', type=str, help='Path of ImageNet data')
args = parser.parse_args()

# ../../../../../dtu/imagenet/ILSVRC/Data/CLS-LOC/
dataPath = args.path

# Define transforms
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the ImageNet dataset
train_data = ImageNet(root=f'{dataPath}train', split='train', transform=transform_train)
val_data = ImageNet(root=f'{dataPath}val', split='val', transform=transform_test)
test_data = ImageNet(root=f'{dataPath}test', split='val', transform=transform_test)

# Define the ResNet-50 model
class ResNet(pl.LightningModule):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.resnet50 = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=False)
        self.resnet50.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.resnet50(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('val_loss', loss)
        self.log('val_acc', pl.metrics.functional.accuracy(y_hat, y))
        
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        return [optimizer], [scheduler]


# Log experiment with WandB
wandb_logger = pl.loggers.WandbLogger(project='bachelor-juca')

# Define the PyTorch Lightning trainer
trainer = pl.Trainer(
    gpus=1, 
    max_epochs=50, 
    progress_bar_refresh_rate=20, 
    logger=wandb_logger
)


# Create data loaders
train_loader = DataLoader(train_data, batch_size=64, num_workers=8, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=64, num_workers=8, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=64, num_workers=8, pin_memory=True)

# Initialize the ResNet model
resnet_model = ResNet()

# Train the model
trainer.fit(resnet_model, train_loader, val_loader)

# Evaluate the model on the test set
trainer.test(resnet_model, test_loader)
