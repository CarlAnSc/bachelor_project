import numpy as np
import pandas as pd
import os
import shutil
import argparse
import torch


def TopFactor_dataset(args):
    """
    This script creates the TopFactor dataset, and the MetalabelIntegration dataset.
    Inputs needed are the imagenet_train_path, imagenet_val_path, and output_path.
    The train path should contain all 1000 classes, and the val path should contain the 50000 images from the ImageNet validation set.
    """
    print("Creating TopFactor dataset")
    annotations_path = "data/annotations/"
    train_root = args.imagenet_train_path
    val_root = args.imagenet_val_path
    output = args.output_path

    val_annotations_top = pd.read_json(
        annotations_path + "imagenet_x_val_top_factor.jsonl", lines=True
    )
    train_annotations_top = pd.read_json(
        annotations_path + "imagenet_x_train_top_factor.jsonl", lines=True
    )

    val_error_rows = [17758, 29800]
    train_error_rows = [1760, 2197, 5087, 6654, 10510, 10960]

    val_annotations_top = val_annotations_top.drop(val_error_rows).reset_index(
        drop=True
    )
    train_annotations_top = train_annotations_top.drop(train_error_rows).reset_index(
        drop=True
    )

    val_filenames = pd.DataFrame(val_annotations_top["file_name"])
    train_filenames = pd.DataFrame(train_annotations_top["file_name"])

    val_classes = pd.from_dummies(
        val_annotations_top.drop(
            columns=["file_name", "class", "justification", "one_word"]
        )
    )
    train_classes = pd.from_dummies(
        train_annotations_top.drop(
            columns=["file_name", "class", "justification", "one_word"]
        )
    )

    val_annotations_top = val_filenames.join(val_classes)
    train_annotations_top = train_filenames.join(train_classes)

    val_annotations_top.columns = ["file_name", "class"]
    train_annotations_top.columns = ["file_name", "class"]
    train_annotations_top["nclass"] = train_annotations_top["file_name"].apply(
        lambda x: x.split("_")[0]
    )
    os.makedirs(output + "TopFactor/val/")
    os.makedirs(output + "TopFactor/train/")

    idx_to_class = {
        0: "multiple_objects",
        1: "background",
        2: "color",
        3: "brighter",
        4: "darker",
        5: "style",
        6: "larger",
        7: "smaller",
        8: "object_blocking",
        9: "person_blocking",
        10: "partial_view",
        11: "pattern",
        12: "pose",
        13: "shape",
        14: "subcategory",
        15: "texture",
    }
    class_to_idx = {v: k for k, v in idx_to_class.items()}

    [os.makedirs(f"{output}TopFactor/val/{key}") for key in class_to_idx.keys()]
    [os.makedirs(f"{output}TopFactor/train/{key}") for key in class_to_idx.keys()]

    print("Copying TopFactor Train images")
    for i, item in val_annotations_top.iterrows():
        i_name = item[0]
        i_class = item[1]

        src = f"{val_root}/{i_name}"
        dest = f"{output}TopFactor/train/{i_class}/{i_name}"

        shutil.copy(src, dest)

    print("Copying TopFactor Val images")
    for i, item in train_annotations_top.iterrows():
        i_name = item[0]
        i_class = item[1]
        i_n_class = item[2]

        src = f"{train_root}/{i_n_class}/{i_name}"
        dest = f"{output}TopFactor/val/{i_class}/{i_name}"

        shutil.copy(src, dest)


def MetaLabelIntegration_dataset(args):
    print("Creating MetalabelIntegration dataset")
    annotations_path = "data/annotations/"
    train_root = args.imagenet_train_path
    val_root = args.imagenet_val_path
    output = args.output_path
    val_wnids = torch.load(annotations_path + "meta.bin")[1]

    val_images = sorted(os.path.join(val_root, image) for image in os.listdir(val_root))
    for wnid in set(val_wnids):
        os.makedirs(os.path.join(output, "MetalabelIntegration", "train", wnid))
        os.makedirs(os.path.join(output, "MetalabelIntegration", "val_not_used", wnid))

    output = os.path.join(output, "MetalabelIntegration")

    val_annotations_top = pd.read_json(
        annotations_path + "imagenet_x_val_top_factor.jsonl", lines=True
    )
    train_annotations_top = pd.read_json(
        annotations_path + "imagenet_x_train_top_factor.jsonl", lines=True
    )

    print("Copying MetalabelIntegration Train images")
    for wnid, img in zip(val_wnids, val_images):
        if os.path.basename(img) in val_annotations_top["file_name"].values:
            shutil.copy(img, os.path.join(output, "train", wnid, os.path.basename(img)))

    print("Copying MetalabelIntegration Val images")
    for _, row in train_annotations_top.iterrows():
        wnid = row["file_name"].split("_")[0]
        img = os.path.join(train_root, wnid, row["file_name"])
        shutil.copy(
            img, os.path.join(output, "val_not_used", wnid, os.path.basename(img))
        )


def main(args):
    TopFactor_dataset(args)
    MetaLabelIntegration_dataset(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup data for this bachelor project")
    parser.add_argument(
        "--imagenet_val_path",
        type=str,
        help="Path to folder containing raw ImageNet val data, 50000 pictures",
        default="none",
    )
    parser.add_argument(
        "--imagenet_train_path",
        type=str,
        help="Path to folder containing raw ImageNet train data in 1000 folders",
        default="none",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to folder where datasets should be placed",
        default="data",
    )
    args = parser.parse_args()
    main(args)
