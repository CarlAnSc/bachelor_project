import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


df2 = pd.read_csv('../../metalabel_objectivity/transformed/scoresFiles.csv',index_col=0)

def relavant_images():
    billeder_svær = []
    billeder_nem = []

    Images2 = df2.groupby('file').mean()[['accuracy', 'balacc', 'f1score', 'jaccard']]
    for i in range(len(list(Images2.columns))):
        k = np.argmin( Images2.iloc[:,i]) 
        billeder_svær.append(Images2.index[k])

    for i in range(len(list(Images2.columns))):
        p = np.argmax( Images2.iloc[:,i]) 
        billeder_nem.append(Images2.index[p])

    set_svær = list(set(billeder_svær))
    set_nem = list(set(billeder_nem))
    return set_nem, set_svær



def get_sample(set_nem,set_svær):

    df1 = pd.read_csv('../../metalabel_objectivity/df_30img_samples.csv')

    samples = df1[df1['file_name'].isin(set_svær+set_nem)]
    samples = samples.reset_index()

    samplescols = samples.columns[1:17]
    k = samples.iloc[:, 1:17].values
    samples_labels = []
    l1 = []
    for item in k:
        l1 = []
        for i, thing in enumerate(item):
            if thing == 1:
                l1.append(samplescols[i])
        samples_labels.append(l1)

    samples['labels'] = samples_labels

    root = "../../labeling_web_app/Images/anno_imgs"

    l1 = {}
    for path, subdirs, files in os.walk(root):
        for name in files:
            l1[name] = os.path.join(path, name)
    #print(l1)
    #print(samples['file_name'])
    samples['path'] = samples['file_name'].apply(lambda x: l1[x])
    
    return samples


set_nem, set_svær = relavant_images()
samples = get_sample(set_nem,set_svær)

colors = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]



fig, axes = plt.subplots(2, 2, figsize=(10, 8))
dict_x = {0: r'$\bf{\theta}$: 76%   $\bf{\theta_{bal}}$: 46%   $\bf{\kappa}$: 6%    $\bf{J}$: 4%',
           1: r'$\bf{\theta}$: 95%   $\bf{\theta_{bal}}$: 91%   $\bf{\kappa}$: 65%    $\bf{J}$: 54%' ,
             2: r'$\bf{\theta}$: 63%   $\bf{\theta_{bal}}$: 57%   $\bf{\kappa}$: 40%    $\bf{J}$: 24%',
               3:r'$\bf{\theta}$: 91%   $\bf{\theta_{bal}}$: 65%   $\bf{\kappa}$: 74%    $\bf{J}$: 62%'}
for i, ax in enumerate(axes.flat):
    ax.set_xlabel(dict_x[i])
    img = plt.imread(samples.iloc[i]['path'])
    ax.imshow(img)
    ax.set_title(samples['labels'][i], fontsize=10)
    props = dict(boxstyle='round', facecolor='cyan', alpha=0.6)
    ax.text(0.05, 0.95, samples.iloc[i]['str_label'], transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)
    #ax.axis('off')
    ax.tick_params(left=False, labelleft=False, right=False, bottom=False, labelbottom=False)
    if i==0:
        ax.text(x=110,y=-60, s='Disaggreement', fontsize=18, fontweight='bold', color=colors[4])
    if i==1:
        ax.text(x=40,y=-60, s='Aggreement', fontsize=18, fontweight='bold', color=colors[1])

#fig.tight_layout()
#plt.show()

fig.savefig('../../figures/Disagreement_plot.png')