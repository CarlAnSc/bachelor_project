import matplotlib.ticker as mtick
import pandas as pd
import matplotlib.pyplot as plt



path = '../../data/annotations/'
dfF = pd.read_csv(path + 'filename_label.csv')
dfL = pd.read_csv(path + 'imagenet_labels.txt', header=None)

LoL = dfL[1].to_numpy()
dfF['str_label'] = dfF['label'].apply(lambda x : LoL[x])
dfF
dftrain_multi = pd.read_json(path + 'imagenet_x_train_multi_factor.jsonl', lines=True)
dftrain_multi['str_label'] = dftrain_multi['class'].apply(lambda x : LoL[x])


N = len(dftrain_multi['str_label'].value_counts())
colors = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]

fig, axs = plt.subplots(1,2,figsize=(16, 4))

cate =  [['maillot', 'tape player', 'horned viper', 'Eskimo dog', 'monitor',
       'cassette player'],[ 'llama', 'coucal', 'convertible', 'bullet train',
       'trilobite', 'king penguin'] ]

train_small = dftrain_multi['str_label'].value_counts()[[0,1,2,3,4,5,N-7,N-5,N-4,N-3,N-2,N-1]] #.plot(kind='bar', figsize=(12,4), alpha=0.4, fontsize=20, color=colors[0])
val_large = dfF['str_label'].value_counts()[['maillot', 'tape player', 'horned viper', 'Eskimo dog', 'monitor',
       'cassette player', 'llama', 'coucal', 'convertible',
       'bullet train', 'trilobite', 'king penguin']]
dfboth = pd.concat([train_small, val_large], axis=1)
for i,ax in enumerate(axs):
    if i == 0:
        ax.set_ylabel('Number of images', fontsize=12)

    ax = dfboth.loc[cate[i]].plot(kind='bar', ax=ax, legend=False, figsize=(12,4), fontsize=12, color=[colors[1],colors[0]], rot=20)
    
fig.legend(['Training set multiclass', 'Validation set multiclass'], fontsize=10, loc=9)
fig.tight_layout()
fig.subplots_adjust(top=0.87)
plt.savefig('../../figures/1-multiclass_distribution.png', bbox_inches= 'tight')