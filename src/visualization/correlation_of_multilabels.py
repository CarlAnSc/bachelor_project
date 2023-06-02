import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


path = "data/annotations/"
dftrain_multi = pd.read_json(path + "imagenet_x_train_multi_factor.jsonl", lines=True)
dfval_multi = pd.read_json(path + "imagenet_x_val_multi_factor.jsonl", lines=True)
dftrain_multi_ = dftrain_multi.iloc[:, 2:18]
dfval_multi_ = dfval_multi.iloc[:, 2:18]
df_all = pd.concat([dftrain_multi_, dfval_multi_])

df_all.columns = df_all.columns.str.replace("_", " ")
# print(np.cov(df_all.pose, df_all.background) / (np.std(df_all.pose) * np.std(df_all.background)) )
df_all = df_all[
    [
        "pose",
        "partial view",
        "object blocking",
        "person blocking",
        "multiple objects",
        "smaller",
        "larger",
        "brighter",
        "darker",
        "background",
        "color",
        "shape",
        "texture",
        "pattern",
        "style",
        "subcategory",
    ]
]
Corr = df_all.corr()

mask1 = Corr.abs() < 0.1
mask2 = np.triu(np.ones_like(Corr, dtype=bool))
mask = mask1 | mask2


cmap = sns.diverging_palette(240, 150, as_cmap=True)


sns.heatmap(
    Corr,
    annot=False,
    fmt=".2f",
    cmap=cmap,
    vmin=0.0,
    vmax=0.4,
    center=0,
    square=False,
    linewidths=0.5,
    cbar_kws={"shrink": 0.5},
    mask=mask2,
)
plt.tight_layout()
plt.xticks(rotation=90)
plt.savefig("figures/correlation_multilabels.png", bbox_inches="tight")
