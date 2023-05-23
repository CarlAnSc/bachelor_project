import argparse
import torch
import pytorch_lightning as pl
import os
import numpy as np
from dotenv import load_dotenv, find_dotenv
from torch.utils.data import DataLoader, RandomSampler
from src.data.dataloader import UseMetaData, ValTransforms
from src.models.ResNet50_use_meta import ResNet_withMeta
from torch.utils.data.sampler import SubsetRandomSampler
import pdb

def main(args):
    
    dotenvpath = find_dotenv()
    load_dotenv(dotenvpath)
    os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
    # Set seed
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision('medium')

    # Log experiment with WandB
    wandb_logger = pl.loggers.WandbLogger(project="bachelor-juca")
    # Set args:
    wandb_logger.experiment.config.update(args)
    # Define the PyTorch Lightning trainer
    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_acc1", mode="max"
    )
    trainer = pl.Trainer(
        accelerator="auto", max_epochs=args.epochs, logger=wandb_logger, callbacks=[model_checkpoint_callback]
    )
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
        sampler = train_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        train_data,
        sampler = val_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Initialize the ResNet model
    resnet_model = ResNet_withMeta(args, num_classes=number_of_classes)
    # Train the model
    trainer.fit(resnet_model, train_loader, val_loader)


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
    
    args = parser.parse_args()
    main(args)
