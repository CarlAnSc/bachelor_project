from torchvision import datasets, transforms


class TopFactor(datasets.ImageFolder):
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
