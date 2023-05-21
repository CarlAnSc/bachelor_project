from tqdm.auto import tqdm
import torchvision.models.resnet as resnet
import torch
import argparse
import pickle
import numpy as np
import pandas as pd
#from lazypredict.Supervised import LazyClassifier
import sklearn
import xgboost as xgb

from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from mlxtend.evaluate import mcnemar_table
from mlxtend.evaluate import mcnemar

classifier_dict = {'k-nearest': KNeighborsClassifier, 'logistic-regr': LogisticRegression, 'Xgb': xgb.XGBClassifier}


def test_classifier_fold(classifier, X_train_cat, y_train_cat, X_test_cat, y_test_cat, X_train_img, y_train_img, X_test_img, y_test_img):

    clf_cat = classifier()
    clf_img = classifier()

    clf_cat.fit(X_train_cat, y_train_cat)
    clf_img.fit(X_train_img, y_train_img)

    y_pred_cat = clf_cat.predict(X_test_cat)
    y_pred_img = clf_img.predict(X_test_img)

    accuracy_cat = sklearn.metrics.accuracy_score(y_test_cat, y_pred_cat)
    accuracy_img = sklearn.metrics.accuracy_score(y_test_img, y_pred_img)

    print('Accuracy cat: ', sklearn.metrics.accuracy_score(y_test_cat, y_pred_cat))
    print('Accuracy img: ', sklearn.metrics.accuracy_score(y_test_img, y_pred_img))

    return clf_cat, clf_img, y_pred_cat, y_pred_img, accuracy_cat, accuracy_img


def main(args):

    CLASSIFIER = classifier_dict[args.classifier]

    val_dict = pickle.load(open('../../data/train_embeddings.pkl', 'rb')) # De 50.000 billeder


    val_img_data = []
    val_meta_data = []
    val_labels = []

    for i in val_dict.keys():
        val_img_data.append(torch.from_numpy(val_dict[i][0]))
        val_meta_data.append(torch.from_numpy(val_dict[i][1]))
        val_labels.append(torch.from_numpy(val_dict[i][2]))


    val_img_data = torch.cat(val_img_data, 0).detach()
    val_meta_data = torch.cat(val_meta_data, 0)
    val_cat_data = torch.cat([val_img_data, val_meta_data], 1).numpy()
    val_labels = torch.cat(val_labels, 0).numpy()


    X = val_cat_data

    N_splits = 5
    kf = KFold(n_splits=N_splits, shuffle=True, random_state=42)
    kf.get_n_splits(X)

    print(kf)

    df_acc = pd.DataFrame(columns=['Using Meta', 'Just images', 'Delta', 'McNemar p-value'])
    M_table_cumm = np.zeros((2,2))
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        print(f"Fold {i + 1}:")
        _,_, y_pred_cat, y_pred_img, acc_cat, acc_img = test_classifier_fold(classifier=CLASSIFIER,
                                                        X_train_cat = val_cat_data[train_index],
                                                            y_train_cat = val_labels[train_index],
                                                            X_test_cat = val_cat_data[test_index],
                                                                y_test_cat = val_labels[test_index], 
                                                                X_train_img = val_img_data[train_index],
                                                                    y_train_img = val_labels[train_index],
                                                                    X_test_img = val_img_data[test_index],
                                                                        y_test_img = val_labels[test_index])
        # Run a mcnemar test for the two classifiers
        M_table = mcnemar_table(y_target= val_labels[test_index],
                                y_model1 = y_pred_cat,
                                y_model2 = y_pred_img)
        M_table_cumm += M_table
        chi2, p = mcnemar(ary=M_table, corrected=False)
        # append accuracies to df_acc
        df_acc.loc[i] = [acc_cat, acc_img, acc_cat- acc_img, p]

    print('------------Cummulated McNemar test------------')
    CI, p_value = mcnemar(ary=M_table_cumm, corrected=False) 
    df_acc['Final CI'] = CI
    df_acc['Final p'] = p_value
    print(df_acc)
    df_acc.to_csv(f'../../k-fold/{args.classifier}_mcnemar.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Doing McNemartest for k-fold")
    parser.add_argument("--classifier", type=str, help="The type of classifier")
    args = parser.parse_args()
    main(args)
