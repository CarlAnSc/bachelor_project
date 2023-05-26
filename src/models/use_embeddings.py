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
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from mlxtend.evaluate import mcnemar_table
from mlxtend.evaluate import mcnemar
import scipy.stats as st

from time import time

classifier_dict = {'k-nearest': KNeighborsClassifier, 'logistic-regr': LogisticRegression, 'Xgb': xgb.XGBClassifier, 'rfc': RandomForestClassifier , 'svc':  SVC, 'Lsvc': LinearSVC}


def mcnemar_ML(Matrix, alpha=0.05):
    """ Code from the toolBox from course 02450 (Introduction to Machine Learning and Data mining
     However, it has been modified to fit the purpose of this project. """

    # perform McNemars test

    n = sum(Matrix.flat)
    n12 = Matrix[0,1]
    n21 = Matrix[1,0]

    #thetahat = (n12-n21)/n
    thetahat = (n21-n12)/n

    Etheta = thetahat

    Q = n**2 * (n+1) * (Etheta+1) * (1-Etheta) / ( (n*(n12+n21) - (n12-n21)**2) )
    #Q = n**2 * (n+1) * (Etheta+1) * (1-Etheta) / ( (n*(n12+n21) - (n21-n12)**2) )

    p = (Etheta + 1)*0.5 * (Q-1)
    #p = (1-Etheta)*0.5 * (Q-1)
    q = (1-Etheta)*0.5 * (Q-1)
    #q = (Etheta + 1)*0.5 * (Q-1)

    CI = tuple(lm * 2 - 1 for lm in st.beta.interval(1-alpha, a=p, b=q) )

    p = 2*st.binom.cdf(min([n12,n21]), n=n12+n21, p=0.5)
    print("Result of McNemars test using alpha=", alpha)
    print("Comparison matrix n")
    print(Matrix)
    if n12+n21 <= 10:
        print("Warning, n12+n21 is low: n12+n21=",(n12+n21))

    print(f"Approximate 1-alpha confidence interval of theta: [thetaL,thetaU] ={CI}")
    print(f"p-value for two-sided test A and B have same accuracy (exact binomial test): p={p}")

    return thetahat, CI, p


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


    isPCA = ''
    if args.pca:
        filename = '../../data/embeddings_PCA.pickle'
        dict = pickle.load(open(filename, 'rb'))
        val_img_data = dict['img']
        val_cat_data = dict['cat']
        val_labels = dict['labels']
        isPCA = 'PCA'

    if args.pcaULTRA:
        filename = '../../data/embeddings_PCA-ULTRA.pickle'
        dict = pickle.load(open(filename, 'rb'))
        val_img_data = dict['img']
        val_cat_data = dict['cat']
        val_labels = dict['labels']
        isPCA = 'PCAULTRA'
    
    if args.tsne:
        filename = '../../data/embeddings_TSNE.pickle'
        dict = pickle.load(open(filename, 'rb'))
        val_img_data = dict['img']
        val_cat_data = dict['cat']
        val_labels = dict['labels']
        isPCA = 'TSNE'

    else: 
        filename = '../../data/train_embeddings.pkl'
        val_dict = pickle.load(open(filename, 'rb')) # De 50.000 billeder


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
    
    startT = time()

    X = val_cat_data

    N_splits = 5
    kf = KFold(n_splits=N_splits, shuffle=True, random_state=42)
    kf.get_n_splits(X)

    print(kf)

    df_acc = pd.DataFrame(columns=['Using Meta', 'Just Images', 'Delta', 'Conf' , 'McNemar p-value'])
    M_table_cumm = np.zeros((2,2))
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        if i == 4:

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
            pd.DataFrame(zip(y_pred_img,y_pred_cat, val_labels[test_index]), columns=['base pred', 'meta pred', 'true label']).to_csv(f'../../k-fold/{args.classifier}_{isPCA}-FOLD{i}.csv')
            # Run a mcnemar test for the two classifiers
            M_table = mcnemar_table(y_target= val_labels[test_index],
                                    y_model1 = y_pred_img,
                                    y_model2 = y_pred_cat)
            M_table_cumm += M_table
            thetahat_1, CI_1, p_value_1 = mcnemar_ML(Matrix=M_table,  alpha=0.05)
            # append accuracies to df_acc
            df_acc.loc[i] = [acc_cat, acc_img, thetahat_1, str(CI_1), p_value_1]

    slutT = time()
    print('------------Cummulated McNemar test------------')
    _, p_value_1 = mcnemar(ary=M_table_cumm, corrected=False)
    thetahat_2, CI_2, p_value_2 = mcnemar_ML(Matrix=M_table_cumm, alpha=0.05)

    df_acc['Final acc_using_meta'] = df_acc['Using Meta'].mean()
    df_acc['Final acc_just_images'] = df_acc['Just Images'].mean()

    df_acc['Final p_1'] = p_value_1
    df_acc['Final theta_2'] = thetahat_2
    df_acc['Final CI_2'] = str(CI_2)
    df_acc['Final p_2'] = p_value_2
    

    print(df_acc)
    df_acc.to_csv(f'../../k-fold/{args.classifier}_mcnemar{isPCA}.csv')

    print(f'-----------------------------{slutT-startT}--------------------------')
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Doing McNemartest for k-fold")
    parser.add_argument("--classifier", type=str, help="The type of classifier")
    parser.add_argument("--pca", type=str, help="use pca data")
    parser.add_argument("--pcaULTRA", type=str, help="use ultra small pca data")
    parser.add_argument("--tsne", type=str, help="use tsne data")



    args = parser.parse_args()
    main(args)
