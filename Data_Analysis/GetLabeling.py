import pandas as pd
from scipy import spatial
from sklearn.metrics import jaccard_score
import numpy as np



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


def Tests(dataframe: pd.DataFrame, dataframeTrue: pd.DataFrame):

    userScores = {'cosine': [], 'jaccard': []}
    for _, row in dataframeTrue.iterrows():
        #COCKERSPANIER
        try:
            file, labels = row['file_name'], row[categories]
            #Get labels in correct order
            trueLabels = labels[categories].values
            print(file)
            userLabels = dataframe[dataframe.file_name == file][categories].values[0]
            print(userLabels, '\n', trueLabels)


            # Cosine and Jaccard
            cos_sim = 1 - spatial.distance.cosine(trueLabels, userLabels)
            jac_score = jaccard_score(trueLabels.tolist(), userLabels.tolist())

            userScores['cosine'].append(cos_sim)
            userScores['jaccard'].append(jac_score)

            # F1 på categories

        except:
            print('Something is wrong')

    return userScores



if __name__ == "__main__":

    df = ToDataframe('../Julius.csv')
    dfTrue = TrueData()

    Julius = Tests(df,dfTrue)

    print(np.mean(Julius['cosine']))
    print(np.mean(Julius['jaccard']))
    