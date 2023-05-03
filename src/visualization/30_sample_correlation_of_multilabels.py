
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


path = '../../metalabel_objectivity/transformed/'
dfall = pd.read_csv(path + 'allData.csv')
#dfval_multi = pd.read_json(path + 'imagenet_x_val_multi_factor.jsonl', lines=True)
dfall = dfall.iloc[:,2:18]
#dfval_multi_ = dfval_multi.iloc[:,2:18]
#df_all = pd.concat([dftrain_multi_, dfval_multi_])
dfall.columns = dfall.columns.str.replace('_', ' ')
print(dfall.columns)

Corr = dfall.corr()

mask1 = Corr.abs() < 0.1
mask2 = np.triu(np.ones_like(Corr, dtype=bool))
mask = mask1 | mask2
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(Corr, annot=False, fmt='.2f', cmap=cmap, vmin=0.0, vmax=0.4, center=0, square=False, linewidths=.5, cbar_kws={"shrink": .5}, mask=mask2)
plt.tight_layout()
plt.xticks(rotation=90) 
plt.savefig('../../figures/30samples_correlation_multilabels.png', bbox_inches= 'tight')