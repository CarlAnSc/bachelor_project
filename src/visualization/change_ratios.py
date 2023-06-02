import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.model_selection import KFold
import pickle
import torch


list_labels = [
    "multiple objects",
    "background",
    "color",
    "brighter",
    "darker",
    "style",
    "larger",
    "smaller",
    "object blocking",
    "person blocking",
    "partial view",
    "pattern",
    "pose",
    "shape",
    "subcategory",
    "texture",
]


def ratio_of_occurence(filename):
    df = pd.read_csv(filename, index_col=0)

    helping_index = (df["meta pred"] == df["true label"]) & (
        df["meta pred"] != df["base pred"]
    )
    harming_index = (df["base pred"] == df["true label"]) & (
        df["meta pred"] != df["base pred"]
    )
    # change_index = (df['meta pred'] != df['base pred']) & (df['meta pred'] != df['true label']) & (df['base pred'] != df['true label'])
    change_index = df["meta pred"] != df["base pred"]

    df_helping = df[helping_index]
    df_harming = df[harming_index]
    df_change = df[change_index]

    series_helper = df_metalabels[helping_index].mean() / df_metalabels.mean()
    series_harmer = df_metalabels[harming_index].mean() / df_metalabels.mean()
    series_change = df_metalabels[change_index].mean() / df_metalabels.mean()
    # series_change1 = df_metalabels[change_index1].mean() / df_metalabels.mean()

    df_ratios = pd.DataFrame(
        [series_change, series_helper, series_harmer],
        index=["change", "helper", "disturber"],
    )

    # print(filename)
    # print('helping',len(df_helping), df_metalabels[helping_index].sum(), '\n')
    # print('disturbing', len(df_harming),df_metalabels[harming_index].sum(), '\n')
    # print('change', len(df_change), df_metalabels[change_index].sum())

    # print('')
    return df_ratios, [
        df_metalabels[change_index].sum(),
        df_metalabels[helping_index].sum(),
        df_metalabels[harming_index].sum(),
    ]


def make_plot(df_ratios, classifier_name, significant):
    colors = ["silver", "mediumaquamarine", "lightcoral", "#f4a261", "#e76f51"]

    fig, ax = plt.subplots(1, 1, figsize=(16, 4))
    # set y-axis limit
    g = (df_ratios.T - 1).plot(
        kind="bar",
        ax=ax,
        legend=False,
        figsize=(12, 4),
        fontsize=10,
        color=colors[:4],
        rot=40,
        bottom=1,
    )
    # ax.errorbar(data=df_std, x='Group', y='Mean', yerr='SD', ls='', lw=3, color='black')
    # sns.barplot(data=(df_ratios.T - 1),ax=ax, legend=False, figsize=(12,4), fontsize=10, color=colors[:3], rot=40, bottom=1, errorbar=('ci', 95))
    # xlim = g.get_xlim()
    # plt.errorbar(x, y, yerr=yerr, linewidth=0, elinewidth=1.8, alpha=1, c=".35")
    for i, c in enumerate(g.containers):
        for j, bar in enumerate(c):
            if significant[i][j] < 10:
                bar.set_alpha(0.3)

    leg = ax.legend(
        ["change", "helping", "disturbing"], fontsize=10, loc="upper center"
    )  # , labelcolor=colors[:3])
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    ax.hlines(y=1, xmin=-1, xmax=16, linewidth=2, color="dimgray")
    ax.set_ylim(0, 3)
    # set x-axis at value y = 1

    # ax.spines['left'].set_position('center')
    # ax.spines['bottom'].set_position('center')
    # Eliminate upper and right axes
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")

    plt.rcParams.update({"font.size": 18})
    plt.xticks(rotation=40, ha="right", fontsize=10)
    plt.title(f"Ratio of occurence of meta labels ({classifier_name})", fontsize=20)
    plt.savefig(
        f"figures/ratio_of_occurence_{classifier_name}.png", bbox_inches="tight"
    )
    plt.plot()


# init main
if __name__ == "__main__":
    val_dict = pickle.load(
        open("data/train_embeddings.pkl", "rb")
    )  # De 50.000 billeder
    val_img_data = []
    val_meta_data = []
    val_labels = []

    for i in val_dict.keys():
        val_img_data.append(torch.from_numpy(val_dict[i][0]))
        val_meta_data.append(torch.from_numpy(val_dict[i][1]))
        val_labels.append(torch.from_numpy(val_dict[i][2]))
    # First get the correct indexes for the k-split

    val_img_data = torch.cat(val_img_data, 0).detach()
    val_meta_data = torch.cat(val_meta_data, 0)
    val_cat_data = torch.cat([val_img_data, val_meta_data], 1).numpy()
    val_labels = torch.cat(val_labels, 0).numpy()

    N_splits = 5
    kf = KFold(n_splits=N_splits, shuffle=True, random_state=42)
    kf.get_n_splits(val_meta_data)

    list_of_meta_labels_in_kfold = []

    for i, (train_index, test_index) in enumerate(kf.split(val_meta_data)):
        # convert to numpy array
        X_train = np.array(val_meta_data[test_index])
        list_of_meta_labels_in_kfold.append(X_train)
    list_of_meta_labels_in_kfold
    meta_label_ready = np.vstack(list_of_meta_labels_in_kfold)
    meta_label_ready.shape

    df_metalabels = pd.DataFrame(
        meta_label_ready,
        columns=[
            "multiple objects",
            "background",
            "color",
            "brighter",
            "darker",
            "style",
            "larger",
            "smaller",
            "object blocking",
            "person blocking",
            "partial view",
            "pattern",
            "pose",
            "shape",
            "subcategory",
            "texture",
        ],
    )

    for name in ["knn", "log", "svc", "rfc", "Lsvc", "xgb"]:
        df_ratios, signi = ratio_of_occurence(
            "results/use_metalabels_kfold/df{}.csv".format(name)
        )
        make_plot(df_ratios, name, signi)
