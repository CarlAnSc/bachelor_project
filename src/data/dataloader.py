from torchvision import datasets, transforms
import torch.utils.data as data
import pandas as pd
import os
import torch


class TopFactor(datasets.ImageFolder):
    """
    TopFactor dataset, just uses ImageFolder on folder path, that is split into class folders.
    """

    def __init__(self, folder_path, *args, **kwargs):
        super().__init__(folder_path, *args, **kwargs)


def ValTransforms():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )


class MultiLabel(datasets.ImageFolder):
    """
    MultiLabel dataloader for our own TopFactor dataset.
    Takes the path of the data.
    """

    def __init__(self, type_str, folder_path, annotations_path, *args, **kwargs):
        self.type = type_str
        super().__init__(folder_path + self.type, *args, **kwargs)
        if self.type == "val":
            self.label_df = pd.read_json(
                annotations_path + "imagenet_x_" + "train" + "_multi_factor.jsonl",
                lines=True,
            )
        if self.type == "train":
            self.label_df = pd.read_json(
                annotations_path + "imagenet_x_" + "val" + "_multi_factor.jsonl",
                lines=True,
            )

    # multilabel getitem method:
    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        filename = os.path.split(path)[1]
        # print(filename)
        # overwrite multilabel target:
        target = self.label_df[self.label_df.file_name == filename].to_numpy()[0][2:18]
        target = target.astype(float)
        target = torch.from_numpy(target)
        return sample, target
    

class UseMetaData(datasets.ImageFolder):
    """
    Combined dataset for using metalabels with the ImageNet Dataset.
    """

    def __init__(self, type_str, folder_path, annotations_path, *args, **kwargs):
        self.type = type_str
        super().__init__(folder_path + self.type, *args, **kwargs)
        if self.type == "val":
            self.label_df = pd.read_json(
                annotations_path + "imagenet_x_" + "train" + "_multi_factor.jsonl",
                lines=True,
            )
        if self.type == "train":
            self.label_df = pd.read_json(
                annotations_path + "imagenet_x_" + "val" + "_multi_factor.jsonl",
                lines=True,
            )

    # multilabel getitem method:
    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        filename = os.path.split(path)[1]
        # overwrite multilabel target:
        meta_labels = self.label_df[self.label_df.file_name == filename].to_numpy()[0]
        meta_labels = meta_labels[2:18]
        meta_labels = meta_labels.astype(float)
        meta_labels = torch.from_numpy(meta_labels)
        meta_labels = meta_labels.type(torch.FloatTensor)
        target = self.label_df[self.label_df.file_name == filename].to_numpy()[0][1]
        target = int(target)
        return sample, meta_labels, target


class UseMetaData_Sampletraining(datasets.ImageFolder):
    """
    Combined dataset for using metalabels with the ImageNet Dataset.
    """

    def __init__(self, folder_path, *args, **kwargs):
        super().__init__(folder_path, *args, **kwargs)
        
    
    # multilabel getitem method:
    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)


        return sample, target
    
