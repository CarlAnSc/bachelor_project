import tqdm
import torchvision.models.resnet as resnet
import torch
import os
import argparse
from dotenv import load_dotenv, find_dotenv
from src.data.dataloader import UseMetaData, ValTransforms
from torch.utils.data import DataLoader
import csv
import pickle
import numpy as np
from torchmetrics import Accuracy

dotenvpath = find_dotenv()
load_dotenv(dotenvpath)

annotation_path = "data/annotations/"
path = '/mnt/f/MetalabelIntegration/'

train_data = UseMetaData(
        "train", path, annotation_path, transform=ValTransforms()
    )
val_data = UseMetaData("val", path, annotation_path, transform=ValTransforms())
    
number_of_classes = len(train_data.classes)
accuracy = Accuracy('multiclass', num_classes=1000).to('cuda')

train_loader = DataLoader(
        train_data,
        batch_size=8,
        num_workers=8,
    )

val_loader = DataLoader(
        val_data,
        batch_size=8,
        num_workers=8,
    )

model = torch.hub.load(
                "pytorch/vision:v0.9.0",
                "resnet50",
                weights="ResNet50_Weights.IMAGENET1K_V1",
            )

# model.fc = torch.nn.Identity()
model.eval()
device = torch.device('cuda')
model.to(device)

# Train
# train_dict = {}

# for i, batch in enumerate(tqdm.tqdm(train_loader)):
#     train_dict[i] = [model(batch[0].to(device)).cpu().detach().numpy(), batch[1].numpy(), batch[2].numpy()]

# pickle.dump(train_dict, open('/data/train_embeddings.pkl', 'wb'))

# #Val
# val_dict = {}
# for i, batch in enumerate(tqdm.tqdm(val_loader)):
#     val_dict[i] = [model(batch[0].to(device)).cpu().detach().numpy(), batch[1].numpy(), batch[2].numpy()]

# pickle.dump(val_dict, open('val_preds.pkl', 'wb'))

acc = []
for i, batch in enumerate(tqdm.tqdm(val_loader)):
    out = model(batch[0].to(device))
    _, index = torch.max(out, 1)
    print(index)
    print(batch[2])
    print(batch[3])
    acc1 = accuracy(index, batch[2].to(device))
    #print(acc1)
    acc.append(acc1.cpu().detach().numpy())

print(np.mean(acc))