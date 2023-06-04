# Using meta-labeled data to improve deep-learning classification model robustness, and boost data quality

This is the bachelor project of Julius R. og Carl S., at the Technical University of Denmark (DTU), 2023.
The repo contains code from https://github.com/facebookresearch/imagenetx/, as we are recreating a lot of their results.

## Structure of repo:

* `data/` - annotations for data used in the project.
* `dockerfiles/` - the dockerfile for building our webapp.
* `error_ratios/` - our results regarding the ImageNet-X error ratios.
* `figures/` - figures used in the project.
* `hpc_jobs/` - job scripts for our HPC cluster. 
* `labeling_web_app/` - code for the webapp we developed.
* `metalabel_objectivity/` - contains code and results for our subset of annotations of meta labels.
* `results/` - includes some of the key results from the project.
* `src/` - includes all models and code for training and evaluating them.

## Requirements:
You need to download ImageNet on your own. We used the 2012 version, which can be found here: http://image-net.org/download-images.
The validation folder should contain 50,000 images, and the train folder should contain 1000 folders containing all the training images.

Install requirements.txt, with your favorite package manager. We used conda.

Then run the following command, to make the datasets we used. It will take a while.:
    
```python 
python src/data/make_datasets.py --imagenet_val_path PATH_TO_IMAGENET_VAL --imagenet_train_path PATH_TO_IMAGENET_TRAIN --output_path PATH_TO_OUTPUT
```
To use wandb, create an .env file from the .defaultenv file, and fill in your wandb API key.

There are different entrypoints, all located in `src/models/`. An example for training our our MetaLabelIntegration model:

```python
python bachelor_project/src/models/train_model_use_meta.py --path data/MetalabelIntegration/ --epochs 70 --batch_size 8 --lr 1e-5 --weight_decay 0.0005 --momentum 0.9 --optimizer sgd --freeze True
```



## Commands for help:

1. To make docker image of app:

```bash docker build -f dockerfiles/app.dockerfile . -t bach_app:latest```


