import argparse
import torch
import pytorch_lightning as pl
import os
from dotenv import load_dotenv, find_dotenv
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.data.dataloader import MultiLabel, ValTransforms
from src.models.ResNet50_multilabel import ResNetMultiLabel


def main(args):
    dotenvpath = find_dotenv()
    load_dotenv(dotenvpath)
    os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
    # Set seed
    torch.manual_seed(args.seed)
    # Log experiment with WandB
    wandb_logger = pl.loggers.WandbLogger(project="bachelor-juca")
    # Set args:
    wandb_logger.experiment.config.update(args)
    # Define the PyTorch Lightning trainer
    trainer = pl.Trainer(
        accelerator="auto", max_epochs=args.epochs, logger=wandb_logger
    )
    annotation_path = "data/annotations/"

    train_data = MultiLabel(
        "train", args.path, annotation_path, transform=ValTransforms()
    )
    val_data = MultiLabel("val", args.path, annotation_path, transform=ValTransforms())

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    # Initialize the ResNet model

    resnet_model = ResNetMultiLabel(args, num_labels=16)
    # Train the model
    trainer.fit(resnet_model, train_loader, val_loader)

    # Evaluate the model on the test set
    # trainer.test(resnet_model, test_loader)


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
    parser.add_argument(
        "--bootstrap",
        type=bool,
        help="Bootstrap. Will bootstrap with the inverse distribution.",
        default=False,
    )
    parser.add_argument(
        "--weighted_loss",
        type=bool,
        help="Weighted loss. Will weight the loss with the inverse distribution.",
        default=False,
    )
    args = parser.parse_args()
    main(args)
