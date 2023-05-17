from tqdm import tqdm
import torchvision.models.resnet as resnet
import torch
import os
import argparse
from dotenv import load_dotenv, find_dotenv
from src.data.dataloader import UseMetaData, ValTransforms
from torch.utils.data import DataLoader
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lazypredict.Supervised import LazyClassifier
import sklearn
import xgboost

train_dict = pickle.load(open('../../data/train_embeddings.pkl', 'rb'))
val_dict = pickle.load(open('../../data/val_embeddings.pkl', 'rb'))

train_img_data = []
train_meta_data = []
train_labels = []

val_img_data = []
val_meta_data = []
val_labels = []

for i in train_dict.keys():
    train_img_data.append(torch.from_numpy(train_dict[i][0]))
    train_meta_data.append(torch.from_numpy(train_dict[i][1]))
    train_labels.append(torch.from_numpy(train_dict[i][2]))

for i in val_dict.keys():
    val_img_data.append(torch.from_numpy(val_dict[i][0]))
    val_meta_data.append(torch.from_numpy(val_dict[i][1]))
    val_labels.append(torch.from_numpy(val_dict[i][2]))

train_img_data = torch.cat(train_img_data, 0).detach()
train_meta_data = torch.cat(train_meta_data, 0)
train_cat_data = torch.cat([train_img_data, train_meta_data], 1).numpy()
train_labels = torch.cat(train_labels, 0).numpy()

val_img_data = torch.cat(val_img_data, 0).detach()
val_meta_data = torch.cat(val_meta_data, 0)
val_cat_data = torch.cat([val_img_data, val_meta_data], 1).numpy()
val_labels = torch.cat(val_labels, 0).numpy()

classifiers = [('AdaBoostClassifier', sklearn.ensemble._weight_boosting.AdaBoostClassifier),
 ('BaggingClassifier', sklearn.ensemble._bagging.BaggingClassifier),
 ('BernoulliNB', sklearn.naive_bayes.BernoulliNB),
 ('CalibratedClassifierCV', sklearn.calibration.CalibratedClassifierCV),
 ('CategoricalNB', sklearn.naive_bayes.CategoricalNB),
 ('DecisionTreeClassifier', sklearn.tree._classes.DecisionTreeClassifier),
 ('DummyClassifier', sklearn.dummy.DummyClassifier),
 ('ExtraTreeClassifier', sklearn.tree._classes.ExtraTreeClassifier),
 ('ExtraTreesClassifier', sklearn.ensemble._forest.ExtraTreesClassifier),
 ('GaussianNB', sklearn.naive_bayes.GaussianNB),
 ('KNeighborsClassifier',
  sklearn.neighbors._classification.KNeighborsClassifier),
 ('LabelPropagation',
  sklearn.semi_supervised._label_propagation.LabelPropagation),
 ('LabelSpreading', sklearn.semi_supervised._label_propagation.LabelSpreading),
 ('LinearDiscriminantAnalysis',
  sklearn.discriminant_analysis.LinearDiscriminantAnalysis),
 ('LinearSVC', sklearn.svm._classes.LinearSVC),
 ('LogisticRegression', sklearn.linear_model._logistic.LogisticRegression),
 ('NearestCentroid', sklearn.neighbors._nearest_centroid.NearestCentroid),
 ('PassiveAggressiveClassifier',
  sklearn.linear_model._passive_aggressive.PassiveAggressiveClassifier),
 ('Perceptron', sklearn.linear_model._perceptron.Perceptron),
 ('QuadraticDiscriminantAnalysis',
  sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis),
 ('RandomForestClassifier', sklearn.ensemble._forest.RandomForestClassifier),
 ('RidgeClassifier', sklearn.linear_model._ridge.RidgeClassifier),
 ('RidgeClassifierCV', sklearn.linear_model._ridge.RidgeClassifierCV),
 ('SGDClassifier', sklearn.linear_model._stochastic_gradient.SGDClassifier),
 ('StackingClassifier', sklearn.ensemble._stacking.StackingClassifier),
 ('XGBClassifier', xgboost.sklearn.XGBClassifier)]

classifiers1 = [('AdaBoostClassifier', sklearn.ensemble._weight_boosting.AdaBoostClassifier)]

clf = LazyClassifier(verbose=3, ignore_warnings=True, random_state=42, predictions=True, classifiers=classifiers1)

models, predictions = clf.fit(train_cat_data, val_cat_data, train_labels, val_labels)