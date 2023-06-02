import argparse
import torch
import pytorch_lightning as pl
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from torch.utils.data import DataLoader, RandomSampler
from src.data.dataloader import UseMetaData, ValTransforms
from src.models.ResNet50_use_meta import ResNet_withMeta
from torch.utils.data.sampler import SubsetRandomSampler
import pdb
from tqdm import tqdm


def main(args):
    if args.evaluate:
        dotenvpath = find_dotenv()
        load_dotenv(dotenvpath)
        # Set seed
        torch.manual_seed(args.seed)
        torch.set_float32_matmul_precision("medium")

        annotation_path = "bachelor_project/data/annotations/"
        train_data = UseMetaData(
            "train", args.path, annotation_path, transform=ValTransforms()
        )
        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(0.2 * num_train))
        np.random.seed(args.seed)
        np.random.shuffle(indices)
        print(num_train)
        print(split)

        train_idx, val_idx = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        # Create data loaders
        number_of_classes = len(train_data.classes)
        train_loader = DataLoader(
            train_data,
            sampler=train_sampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        val_loader = DataLoader(
            train_data,
            sampler=val_sampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        resnet_model_base = torch.hub.load(
            "pytorch/vision:v0.9.0",
            "resnet50",
            weights="ResNet50_Weights.IMAGENET1K_V1",
        )
        resnet_trained = ResNet_withMeta(
            args, num_classes=number_of_classes
        ).load_from_checkpoint(
            "bachelor_project/checkpoints/epoch=45-step=28106.ckpt",
            args=args,
            num_classes=number_of_classes,
        )

        device = "cuda"
        resnet_model_base.eval()
        resnet_trained.eval()
        resnet_model_base.to(device)
        resnet_trained.to(device)

        base_preds = []
        trained_preds = []
        truth = []

        for i, batch in enumerate(tqdm(val_loader)):
            out_base = resnet_model_base(batch[0].to(device))
            _, preds_base = torch.max(out_base, 1)
            base_preds.append(preds_base.cpu().detach().numpy()[0])

            out_trained = resnet_trained(batch[0].to(device), batch[1].to(device))
            _, preds_trained = torch.max(out_trained, 1)
            trained_preds.append(preds_trained.cpu().detach().numpy()[0])

            truth.append(batch[2].numpy()[0])

        base = base_preds
        trained = trained_preds
        truth = truth

        df = pd.DataFrame([base, trained, truth]).T
        df.columns = ["base", "trained", "truth"]
        print(df)
        df.to_csv("results/use_metalabels_resnet/resnet_preds.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training of model for hpc and local")
    parser.add_argument("--path", type=str, help="Path of ImageNet data")
    parser.add_argument("--lr", type=float, help="Learning rate", default=1e-3)
    parser.add_argument(
        "--weight_decay", type=float, help="Learning rate", default=0.0005
    )
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=50)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=16)
    parser.add_argument("--num_workers", type=int, help="Number of workers", default=8)
    parser.add_argument("--seed", type=int, help="Seed", default=42)
    parser.add_argument(
        "--optimizer",
        type=str,
        help='Choose between the two: "adam" and "sgd" ',
        default="sgd",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        help="Momentum. Will only be used for SGD optimizer, else; ignored.",
        default=0.9,
    )
    parser.add_argument("--freeze", type=bool, help="Freeze layers", default=False)
    parser.add_argument(
        "--evaluate", type=bool, help="Set models to evaluate", default=True
    )

    args = parser.parse_args()
    main(args)
