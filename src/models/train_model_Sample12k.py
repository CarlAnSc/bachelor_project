import argparse
import torch
import pytorch_lightning as pl
import os
from dotenv import load_dotenv, find_dotenv
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.data.dataloader import UseMetaData_Sampletraining, ValTransforms
from src.models.ResNet50 import ResNet
from torchmetrics import Accuracy
import numpy as np
import torchvision
import random


# TODO: Weighted loss
# TODO: Net dataset, 4 classes (background, color, pattern, pose)


def main(args):
    # Define the PyTorch Lightning trainer
    trainer = pl.Trainer(
        accelerator="auto", max_epochs=args.epochs
    )

    train_data = UseMetaData_Sampletraining(args.path, transform=ValTransforms())
    number_of_classes = len(train_data.classes)

    train_loader = DataLoader(
        train_data,
        batch_size=8,
        num_workers=8,
    )
    # _____
    N_dataset = len(train_data.dataset)


    randomlist = random.sample(range(1, N_dataset), 12000)

    trainset_1 = torch.utils.data.Subset(train_data, randomlist)

    trainloader_1 = torch.utils.data.DataLoader(trainset_1, batch_size=64,
                                                num_workers=8)
    # ______
    model = torch.hub.load(
                    "pytorch/vision:v0.9.0",
                    "resnet50",
                    weights="ResNet50_Weights.IMAGENET1K_V1",
                )

# model.fc = torch.nn.Identity()
    model.eval()
    device = torch.device('cuda')
    model.to(device)

    accuracy = Accuracy('multiclass', num_classes=1000).to(device)


    acc = []
    for i, batch in enumerate(trainloader_1):
        out = model(batch[0].to(device))
        _, index = torch.max(out, 1)
        print(index)
        print(batch[2])
        print(batch[3])
        acc1 = accuracy(index, batch[2].to(device))
        #print(acc1)
        acc.append(acc1.cpu().detach().numpy())

    print(np.mean(acc))

    # Evaluate the model on the test set
    # trainer.test(resnet_model, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training of model for hpc and local")
    parser.add_argument("--path", type=str, help="Path of ImageNet data")
    
    args = parser.parse_args()
    main(args)
