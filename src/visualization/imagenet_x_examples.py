import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.imagenet_x import get_factor_accuracies, error_ratio
from src.imagenet_x.utils import load_model_predictions, get_annotation_path
from src.imagenet_x import plots
from src.imagenet_x.evaluate import (
    ImageNetX,
    get_vanilla_transform,
    ImageNetXImageFolder,
    load_annotations,
)
import os
import pickle
from pathlib import Path

df1 = pd.read_csv("metalabel_objectivity/val_imgs_df.csv")
samples = df1.sample(6)

samplescols = samples.columns[1:17]
k = samples.iloc[:, 1:17].values
samples_labels = []
l1 = []
for item in k:
    l1 = []
    for i, thing in enumerate(item):
        if thing == 1:
            l1.append(samplescols[i])
    samples_labels.append(l1)

samples["labels"] = samples_labels

root = "data/ImageNetVal/"

l1 = {}
for path, subdirs, files in os.walk(root):
    for name in files:
        l1[name] = os.path.join(path, name)

samples["path"] = samples["file_name"].apply(lambda x: l1[x])

# plot 6 in a grid with matplotlib cropping them
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
for i, ax in enumerate(axes.flat):
    img = plt.imread(samples.iloc[i]["path"])
    ax.imshow(img)
    ax.set_title(samples_labels[i], fontsize=22)
    props = dict(boxstyle="round", facecolor="cyan", alpha=0.6)
    ax.text(
        0.05,
        0.95,
        samples.iloc[i]["str_label"],
        transform=ax.transAxes,
        fontsize=22,
        verticalalignment="top",
        bbox=props,
    )
    ax.axis("off")
fig.tight_layout()

fig.savefig("figures/imagenet_x_examples.png")
