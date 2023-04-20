import argparse
import torch
import pytorch_lightning as pl
import os
from dotenv import load_dotenv, find_dotenv
from torch.utils.data import DataLoader, RandomSampler
from src.data.dataloader import UseMetaData, ValTransforms
from src.models.ResNet50 import ResNet



def main(args):
    #dotenvpath = find_dotenv()
    #load_dotenv(dotenvpath)
    #os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
    # Set seed
    torch.manual_seed(args.seed)
    # Log experiment with WandB
    #wandb_logger = pl.loggers.WandbLogger(project="bachelor-juca")
    # Set args:
    #wandb_logger.experiment.config.update(args)
    #print(f"WEIGHTED LOSS ER: {args.weighted_loss}")
    # Define the PyTorch Lightning trainer
    trainer = pl.Trainer(
        accelerator="auto", max_epochs=50     #args.epochs    #, logger=wandb_logger
    )

    train_data = UseMetaData(args.path + "train/", transform=ValTransforms())
    val_data = UseMetaData(args.path + "val/", transform=ValTransforms())
    number_of_classes = len(train_data.classes)

        # Create data loaders
        
    sampler = RandomSampler(replacement=True
    )
    train_loader = DataLoader(
        train_data,
        batch_size=64,          #args.batch_size,
        num_workers=8,          #args.num_workers,
        pin_memory=True,
        sampler=sampler,
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=64,           #args.batch_size,
        num_workers=8,          #args.num_workers,
        pin_memory=True,
    )
    # test_loader = DataLoader(test_data, batch_size=32, num_workers=8, pin_memory=True)

    # Initialize the ResNet model
    
    resnet_model = ResNet(args, num_classes=number_of_classes)
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

    args = parser.parse_args()
    main(args)
