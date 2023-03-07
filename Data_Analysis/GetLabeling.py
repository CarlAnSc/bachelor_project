import pandas as pd
from scipy import spatial
from sklearn.metrics import jaccard_score, log_loss, accuracy_score, f1_score
import numpy as np
from os import listdir
import os
from sklearn.metrics import balanced_accuracy_score
import scipy.stats as st



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


def Tests(personName: str, dataframe: pd.DataFrame, dataframeTrue: pd.DataFrame):

    dataframe = dataframe.sort_values(['file_name'])
    dataframeTrue = dataframeTrue.sort_values(['file_name'])

    # All data at ones
    preds = dataframe[categories].to_numpy().flatten()
    trues = dataframeTrue[categories].to_numpy().flatten()

    accuracy = accuracy_score(trues, preds)
    bal_acc = balanced_accuracy_score(trues, preds)
    f1score = f1_score(trues, preds)
    cos_sim = 1 - spatial.distance.cosine(trues, preds)
    jac_score = jaccard_score(trues, preds)
    log_loss1 = log_loss(y_true= trues, y_pred= preds)

    # Seperate data
    personScores = {'person': [personName], 'accuracy': [], 'balacc': [], 'f1score': [], 'cosine': [], 'jaccard': [], 'logloss': []}
    fileScores = {'file': [], 'accuracy': [], 'balacc': [], 'f1score': [], 'cosine': [], 'jaccard': [], 'logloss': [], 'person': [personName]*30}
    cateScores = {'cate': categories, 'accuracy': [],'balacc': [], 'f1score': [], 'cosine': [], 'jaccard': [], 'logloss': [], 'person': [personName]*16}

    personScores['accuracy'].append(accuracy)
    personScores['balacc'].append(bal_acc)
    personScores['f1score'].append(f1score)
    personScores['cosine'].append(cos_sim)
    personScores['jaccard'].append(jac_score)
    personScores['logloss'].append(log_loss1)
    
    
    for _, row in dataframeTrue.iterrows():

        file, labels = row['file_name'], row[categories]
        #Get labels in correct order
        trueLabels = labels[categories].values
        #print(file)
        userLabels = dataframe[dataframe.file_name == file][categories].values[0]
        #print(userLabels, '\n', trueLabels)

        # Cosine and Jaccard and Binary Cross Entropy
        accuracy = accuracy_score(trueLabels.tolist(), userLabels.tolist())
        bal_acc = balanced_accuracy_score(trueLabels.tolist(), userLabels.tolist())
        f1score = f1_score(np.array(trueLabels.tolist()), np.array(userLabels.tolist()))
        cos_sim = 1 - spatial.distance.cosine(trueLabels, userLabels)
        jac_score = jaccard_score(trueLabels.tolist(), userLabels.tolist())
        log_loss1 = log_loss(y_true = trueLabels.astype('int'),y_pred = userLabels.astype('int'))

        fileScores['file'].append(file)
        fileScores['balacc'].append(bal_acc)
        fileScores['accuracy'].append(accuracy)
        fileScores['f1score'].append(f1score)
        fileScores['cosine'].append(cos_sim)
        fileScores['jaccard'].append(jac_score)
        fileScores['logloss'].append(log_loss1)
    
    for column in categories:

        accuracy = accuracy_score(dataframeTrue[column],dataframe[column])
        bal_acc = balanced_accuracy_score(dataframeTrue[column],dataframe[column])
        f1score = f1_score(dataframeTrue[column],dataframe[column])
        cos_sim = 1 - spatial.distance.cosine(dataframeTrue[column],dataframe[column])
        jac_score = jaccard_score(dataframeTrue[column].tolist(), dataframe[column].tolist())
        log_loss1 = log_loss(y_pred = dataframe[column].astype('int'),y_true = dataframeTrue[column].astype('int'), labels= [0,1])
        
        cateScores['accuracy'].append(accuracy)
        cateScores['balacc'].append(bal_acc)
        cateScores['f1score'].append(f1score)
        cateScores['cosine'].append(cos_sim)
        cateScores['jaccard'].append(jac_score)
        cateScores['logloss'].append(log_loss1)
    
    print(len(fileScores['f1score'])   ,    len(cateScores['f1score']))
    return fileScores, cateScores, personScores


def confidence(data: pd.DataFrame):
    list_interval = {}
    for (columnName, columnData) in data.iloc[:,1:-1].items():
        interval = st.t.interval(alpha=0.95,
              df=len(columnData)-1,
              loc=np.mean(columnData), 
              scale=st.sem(columnData))
        list_interval[columnName] = (interval - np.mean(columnData))[1]
    return list_interval

def run_confidence(dfCates, dfFiles, dfPersons):

    # Remove BASELINE
    df1 = dfCates[dfCates.person != 'BASELINE']
    df2 = dfFiles[dfFiles.person != 'BASELINE']
    df3 = dfPersons[dfPersons.person != 'BASELINE']

    # Get confidence interval
    listy1 = [confidence(df1[df1.cate == category]) for category in df1['cate'].unique() ]
    df1Conf = pd.DataFrame.from_records(listy1,index=df1['cate'].unique())

    listy2 = [confidence(df2[df2.file == file]) for file in df2['file'].unique() ]
    df2Conf = pd.DataFrame.from_records(listy2,index=df2['file'].unique())

    dicty3 = confidence(df3)
    df3Conf = pd.DataFrame(dicty3, index=[0])

    df1Conf.to_csv('transformed/confCategories.csv')
    df2Conf.to_csv('transformed/confFiles.csv')
    df3Conf.to_csv('transformed/confPersons.csv')

    #return df1Conf, df2Conf, df3Conf

if __name__ == "__main__":

    # Initialize dataframes
    dfRaw = pd.DataFrame()
    dfFiles = pd.DataFrame()
    dfCates = pd.DataFrame()
    dfPersons = pd.DataFrame()
    dfTrue = TrueData()

    # Set path for annotations
    PATH = 'annotations/'
    files = listdir(PATH)
    # Run through files
    for i,filename in enumerate(files):
        fullfilename = os.path.join(PATH, filename)
        df = ToDataframe(fullfilename)
        df['person'] = [filename[:-4]]*30

        # Get metrics
        dictFiles, dictCate, dictPersons = Tests(filename[:-4], df,dfTrue)

        # Append to each dataframe
        dfRaw = pd.concat([dfRaw,df],ignore_index=False)

        userdf1 = pd.DataFrame.from_dict(dictFiles)
        dfFiles = pd.concat([dfFiles,userdf1],ignore_index=True)

        userdf2 = pd.DataFrame.from_dict(dictCate)
        dfCates = pd.concat([dfCates,userdf2],ignore_index=True)

        userdf3 = pd.DataFrame.from_dict(dictPersons)
        dfPersons = pd.concat([dfPersons,userdf3],ignore_index=True)

    # Get confidence interval and save as csv
    run_confidence(dfCates, dfFiles, dfPersons)

    # Save to csv files
    dfRaw.to_csv('transformed/allData.csv')
    dfFiles.to_csv('transformed/scoresFiles.csv')
    dfCates.to_csv('transformed/scoresCategories.csv')
    dfPersons.to_csv('transformed/scoresPersons.csv')


    