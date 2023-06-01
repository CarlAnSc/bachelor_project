import matplotlib.ticker as mtick
import pandas as pd
import matplotlib.pyplot as plt

all = pd.read_csv('../../metalabel_objectivity/val_imgs_df.csv')
N = len(all)
valimg = pd.read_csv('../../metalabel_objectivity/df_30img_samples.csv')

#Inspired by plots.factor_distribution_comparison()
dfAll = (all.iloc[:,1:17]*100/N).sum()
dfSample = (valimg.iloc[:,1:17]*100/30).sum()
dfko = pd.DataFrame.from_dict({'all': dfAll, 'sample': dfSample})

colors = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]

fig, axs = plt.subplots(1,4,figsize=(18, 4))

cate =  [['pose', 'background', 'pattern', 'color'], [ 'smaller', 'shape', 'partial view', 'subcategory'], ['texture', 'darker', 'style', 'multiple objects'], ['larger', 'object blocking', 'person blocking', 'brighter'] ]
for i,ax in enumerate(axs):
    print(cate[i])
    dfko.index = dfko.index.str.replace('_', ' ')
    ax = dfko.loc[cate[i]].plot.bar(ax=ax,color=[colors[0],colors[1]],width=0.7,rot=20,legend=False)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(), )
    
fig.legend(['Full ImageNet-X validation set','30 randomly selected images for labeling'], loc=9, ncol = 2)


#ax[0].yaxis.set_major_formatter(mtick.PercentFormatter())
plt.gcf().subplots_adjust(bottom=0.15)
plt.savefig('../../figures/1-sample30_distribution.png', bbox_inches= 'tight')
plt.show()
