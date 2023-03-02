import pandas as pd
from scipy import spatial
from sklearn.metrics import jaccard_score, log_loss, accuracy_score, f1_score
import numpy as np
from os import listdir
import os



categories = ['pose', 'partial_view', 'object_blocking', 'person_blocking', 'multiple_objects', 'smaller', 'larger',
                      'brighter', 'darker', 'background', 'color', 'shape', 'texture', 'pattern', 'style', 'subcategory']

def ToDataframe(csvfile: str): 
    """Outputs a dataframe with 18 columns"""
    df = pd.read_csv(csvfile,sep=",") # ../Mappe2.csv
    df = df.drop(['Unnamed: 0'], axis=1)
    df['1'] = df['1'].apply(eval)
    
    dfNew = pd.DataFrame(df['1'].to_list()) * 1
    dfNew.columns = categories
    dfNew['list'] = df['1']
    dfNew.insert(0,'file_name',df['0'])

    return dfNew

def TrueData():
    df = pd.read_csv('df_30img_samples.csv')
    return df.iloc[:,:17]


def Tests(personID: int, dataframe: pd.DataFrame, dataframeTrue: pd.DataFrame):

    dataframe = dataframe.sort_values(['file_name'])
    dataframeTrue = dataframeTrue.sort_values(['file_name'])

    #listCosine = pairwise.cosine_similarity(dataframe[categories], dataframeTrue[categories])
    #print(np.array(listCosine).diagonal())

    fileScores = {'file': [], 'accuracy': [], 'f1score': [], 'cosine': [], 'jaccard': [], 'logloss': [], 'person': [personID]*30}
    cateScores = {'cate': categories, 'accuracy': [],'f1score': [], 'cosine': [], 'jaccard': [], 'logloss': [], 'person': [personID]*16}

    for _, row in dataframeTrue.iterrows():

        file, labels = row['file_name'], row[categories]
        #Get labels in correct order
        trueLabels = labels[categories].values
        #print(file)
        userLabels = dataframe[dataframe.file_name == file][categories].values[0]
        #print(userLabels, '\n', trueLabels)

        # Cosine and Jaccard and Binary Cross Entropy
        accuracy = accuracy_score(trueLabels.tolist(), userLabels.tolist())
        f1score = f1_score(np.array(trueLabels.tolist()), np.array(userLabels.tolist()))
        cos_sim = 1 - spatial.distance.cosine(trueLabels, userLabels)
        jac_score = jaccard_score(trueLabels.tolist(), userLabels.tolist())
        log_loss1 = log_loss(y_true = trueLabels.astype('int'),y_pred = userLabels.astype('int'))

        fileScores['file'].append(file)
        fileScores['accuracy'].append(accuracy)
        fileScores['f1score'].append(f1score)
        fileScores['cosine'].append(cos_sim)
        fileScores['jaccard'].append(jac_score)
        fileScores['logloss'].append(log_loss1)
    
    for column in categories:

        accuracy = accuracy_score(dataframe[column], dataframeTrue[column])
        f1score = f1_score(dataframe[column], dataframeTrue[column])
        cos_sim = 1 - spatial.distance.cosine(dataframe[column],dataframeTrue[column])
        jac_score = jaccard_score(dataframe[column].tolist(), dataframeTrue[column].tolist())
        log_loss1 = log_loss(y_pred = dataframe[column].astype('int'),y_true = dataframeTrue[column].astype('int'), labels= [0,1])
        
        cateScores['accuracy'].append(accuracy)
        cateScores['f1score'].append(f1score)
        cateScores['cosine'].append(cos_sim)
        cateScores['jaccard'].append(jac_score)
        cateScores['logloss'].append(log_loss1)
    
    print(len(fileScores['f1score'])   ,    len(cateScores['f1score']))
    return fileScores, cateScores



if __name__ == "__main__":
    dfRaw = pd.DataFrame()
    dfFiles = pd.DataFrame()
    dfCates = pd.DataFrame()
    dfTrue = TrueData()

    PATH = 'annotations/'
    files = listdir(PATH)
    for i,filename in enumerate(files):
        filename = os.path.join(PATH, filename)
        df = ToDataframe(filename)
        df['person'] = [i]*30
        dfRaw = pd.concat([dfRaw,df],ignore_index=False)

        dictFiles, dictCate = Tests(i, df,dfTrue)

        userdf1 = pd.DataFrame.from_dict(dictFiles)
        dfFiles = pd.concat([dfFiles,userdf1],ignore_index=True)

        userdf2 = pd.DataFrame.from_dict(dictCate)
        dfCates = pd.concat([dfCates,userdf2],ignore_index=True)

    #print(dfRaw)
    dfRaw.to_csv('transformed/allData.csv')
    dfFiles.to_csv('transformed/scoresFiles.csv')
    dfCates.to_csv('transformed/scoresCategories.csv')
    print(files)

    