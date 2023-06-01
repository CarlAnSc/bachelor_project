import matplotlib.ticker as mtick
import pandas as pd
import matplotlib.pyplot as plt


path = '../../data/annotations/'
dftrain = pd.read_json(path + 'imagenet_x_train_top_factor.jsonl', lines=True)
dftrain_multi = pd.read_json(path + 'imagenet_x_train_multi_factor.jsonl', lines=True)
dfval = pd.read_json(path + 'imagenet_x_val_top_factor.jsonl', lines=True)
dfval_multi = pd.read_json(path + 'imagenet_x_val_multi_factor.jsonl', lines=True)


#Inspired by plots.factor_distribution_comparison()
dftrain_ = (dftrain.iloc[:,2:18]*100/len(dftrain)).sum()
dfval_ = (dfval.iloc[:,2:18]*100/len(dfval)).sum()
dftrain_multi_ = (dftrain_multi.iloc[:,2:18]*100/len(dftrain_multi)).sum()
dfval_multi_ = (dfval_multi.iloc[:,2:18]*100/len(dfval_multi)).sum()
dfko = pd.DataFrame.from_dict({ 'our multi val': dftrain_multi_, 'our multi train': dfval_multi_, 'our val': dftrain_, 'our train': dfval_})

colors = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]

fig, axs = plt.subplots(1,4,figsize=(18, 4))

cate =  [['pose', 'background', 'pattern', 'color'], [ 'smaller', 'shape', 'partial view', 'subcategory'], ['texture', 'darker', 'style', 'multiple objects'], ['larger', 'object blocking', 'person blocking', 'brighter'] ]

for i,ax in enumerate(axs):
    print(cate[i])
    dfko.index = dfko.index.str.replace('_', ' ')
    ax = dfko.loc[cate[i]].plot.bar(ax=ax,color=[colors[0],colors[1], colors[2], colors[3]],width=0.7,rot=20,legend=False)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(), )
    
fig.legend(['Validation set multifactor','Training set multifactor', 'Validation set top factor','Traning set top factor'], loc=9, ncol = 2)


#ax[0].yaxis.set_major_formatter(mtick.PercentFormatter())
plt.gcf().subplots_adjust(bottom=0.15)
plt.savefig('../../figures/all_distribution.png', bbox_inches= 'tight')
plt.show()