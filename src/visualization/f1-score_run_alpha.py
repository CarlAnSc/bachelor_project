
import pandas as pd
import matplotlib.pyplot as plt

dict_f1 = {'texture': 0, 'subcategory': 0, 
           'shape': 0, 'pose': 0.95,
           'pattern': 0.49, 'partial view': 0,
           'person blocking': 0, 'object blocking': 0,
           'smaller': 0.07, 'larger': 0,
            'style': 0, 'darker': 0.01,
            'brighter': 0,'color': 0.23,
            'background': 0.91,'multiple objects': 0}

df_f1 = pd.DataFrame(dict_f1,index=[0]).T
df_f1.columns = ['f1-score']
colors = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]


df_f1.plot(kind='bar', legend=False, figsize=(12,3), fontsize=10, color=colors[1], rot=30, bottom=0)
plt.ylabel('F1-score', fontsize=12)
plt.xlabel('Factor', fontsize=12)
plt.title('F1-score for each factor', fontsize=14)
plt.savefig('../../figures/f1-scores_run_alpha.png', bbox_inches= 'tight')
plt.plot()