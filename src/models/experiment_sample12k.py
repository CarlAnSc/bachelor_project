import argparse
import torch
import pytorch_lightning as pl
import os
from dotenv import load_dotenv, find_dotenv
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.data.dataloader import UseMetaData_Sampletraining, ValTransforms
from src.models.ResNet50 import ResNet
from torchmetrics import Accuracy
from tqdm import tqdm
import numpy as np
import torchvision
import random
import pdb


def main(args):
# Define the PyTorch Lightning trainer
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

    train_data = UseMetaData_Sampletraining(args.path, transform=ValTransforms())
    
    number_of_classes = len(train_data.classes)

    train_loader = DataLoader(
        train_data,
        batch_size=8,
        num_workers=8,
    )
    N_dataset = len(train_loader.dataset)
    mean_accs = []
    
    for i in range(10):
        randomlist = random.sample(range(1, N_dataset), 9000)
        trainset_1 = torch.utils.data.Subset(train_data, randomlist)
        trainloader_1 = torch.utils.data.DataLoader(trainset_1, batch_size=16,
                                                num_workers=8)
        acc = []
        for i, batch in enumerate(tqdm(trainloader_1)):
            out = model(batch[0].to(device))
            _, index = torch.max(out, 1)
            acc1 = accuracy(index, batch[1].to(device))
            acc.append(acc1.cpu().detach().numpy())

        print(np.mean(acc))
        mean_accs.append(np.mean(acc))

    np.array(mean_accs).tofile('mean_accs.txt', sep=',')
    # Evaluate the model on the test set
    # trainer.test(resnet_model, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training of model for hpc and local")
    parser.add_argument("--path", type=str, help="Path of ImageNet data")
    
    args = parser.parse_args()
    main(args)
