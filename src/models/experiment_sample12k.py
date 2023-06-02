import argparse
import torch
import pytorch_lightning as pl
import os
from dotenv import load_dotenv, find_dotenv
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.data.dataloader import UseMetaData_Sampletraining, ValTransforms
from src.models.ResNet50 import ResNet
from torchmetrics import Accuracy
from torchvision.models import vit_b_16, efficientnet_v2_s
from tqdm import tqdm
import numpy as np
import torchvision
import random
import pdb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick


def main(args):
    # Define the PyTorch Lightning trainer
    # model = torch.hub.load(
    #     "pytorch/vision:v0.9.0",
    #     "resnet50",
    #     weights="ResNet50_Weights.IMAGENET1K_V1",
    # )
    model = vit_b_16(weights="ViT_B_16_Weights.IMAGENET1K_V1")
    # model = efficientnet_v2_s(weights="EfficientNet_V2_S_Weights.IMAGENET1K_V1")

    model.eval()
    device = torch.device("cuda")
    model.to(device)
    print(model)
    accuracy = Accuracy("multiclass", num_classes=1000).to(device)

    train_data = UseMetaData_Sampletraining(args.path, transform=ValTransforms())

    train_loader = DataLoader(
        train_data,
        batch_size=32,
        num_workers=8,
    )
    N_dataset = len(train_loader.dataset)
    mean_accs = []

    for i in range(1):
        print(f"Iteration {i}")
        acc = []

        if not args.train_only:
            print("Evaluating on random 12000 images from ImageNet Train set")
            randomlist = random.sample(range(1, N_dataset), 12000)
            trainset_1 = torch.utils.data.Subset(train_data, randomlist)
            trainloader_1 = torch.utils.data.DataLoader(
                trainset_1, batch_size=32, num_workers=8
            )
            for i, batch in enumerate(tqdm(trainloader_1)):
                out = model(batch[0].to(device))
                _, index = torch.max(out, 1)
                acc1 = accuracy(index, batch[1].to(device))
                acc.append(acc1.cpu().detach().numpy())
            print(np.mean(acc))

        else:
            print("Only evaluating on ImageNet-X train set")
            for i, batch in enumerate(tqdm(train_loader)):
                out = model(batch[0].to(device))
                _, index = torch.max(out, 1)
                acc1 = accuracy(index, batch[1].to(device))
                acc.append(acc1.cpu().detach().numpy())
                print(np.mean(acc))

        mean_accs.append(np.mean(acc))

    np.array(mean_accs).tofile("mean_accs_effiecientnet.txt", sep=",")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training of model for hpc and local")
    parser.add_argument("--path", type=str, help="Path of ImageNet data")
    parser.add_argument(
        "--train_only", type=bool, help="only use the train part of ImageNet-X"
    )
    args = parser.parse_args()
    main(args)
