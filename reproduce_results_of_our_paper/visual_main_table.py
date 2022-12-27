# -*- coding: utf-8 -*-


import os
from collections import OrderedDict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.transforms import Bbox

import seaborn as sns

from visual_utils import walk_paths, loaddata, organize_df, create_dir


def axis_shift(ax1, x_shift=0.01, y_shift=0):
    position = ax1.get_position().get_points()
    pos_new = position
    pos_new[:, 0] += x_shift
    pos_new[:, 1] += y_shift
    ax1.set_position(Bbox(pos_new))


def compute_improvement(df, col, last=0):
    our = df.iloc[-5:][col]["CIRS"].mean()
    prev = df.iloc[-last:][col]["CIRS w_o CI"].mean()
    print(f"Improvement on [{col}] of last [{last}] count is {(our - prev) / prev}")


def draw(df_metric, ax1, color, marker, name):
    df_metric.plot(kind="line", linewidth=1, ax=ax1, legend=None, color=color, markevery=int(len(df_metric) / 10),
                   fillstyle='none', alpha=.8, markersize=3)
    for i, line in enumerate(ax1.get_lines()):
        line.set_marker(marker[i])

    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)
    plt.grid(linestyle='dashdot', linewidth=0.8)
    ax1.set_ylabel(name, fontsize=10, fontweight=700)


def visual(df_all, save_fig_dir, savename="three"):
    df_all.rename(columns={r"$\text{CV}_\text{M}$": r"CV_M"}, level=1,
                  inplace=True)
    ways = df_all.columns.levels[0]
    metrics = df_all.columns.levels[1]
    methods = df_all.columns.levels[2]

    # fontsize = 11.5

    methods_list = list(set(methods))
    num_methods = len(methods_list)

    colors = sns.color_palette(n_colors=num_methods)
    markers = ["o", "s", "p", "P", "X", "*", "h", "D", "v", "^", ">", "<", "x", "H"][:num_methods]

    color_kv = dict(zip(methods_list, colors))
    marker_kv = dict(zip(methods_list, markers))

    fig = plt.figure(figsize=(10, 15))
    plt.subplots_adjust(wspace=0.3)
    plt.subplots_adjust(hspace=0.3)

    axs = np.empty([len(metrics), len(ways)], dtype=object)
    for col, way in enumerate(ways):
        df = df_all[way]

        color = [color_kv[name] for name in methods]
        marker = [marker_kv[name] for name in methods]

        for row, metric in enumerate(metrics):
            # print(metric, row, col)
            df_metric = df[metric]
            ax1 = plt.subplot2grid((len(metrics), len(ways)), (row, col))
            axs[row, col] = ax1
            draw(df_metric, ax1, color, marker, metric)

    ax_legend = axs[0][1]
    lines, labels = ax_legend.get_legend_handles_labels()
    dict_label = dict(zip(labels, lines))
    dict_label = OrderedDict(sorted(dict_label.items(), key=lambda x: x[0]))

    ax_legend.legend(handles=dict_label.values(), labels=dict_label.keys(), ncol=10,
                     loc='lower left', columnspacing=0.7,
                     bbox_to_anchor=(-0.20, 1.24), fontsize=10.5)

    fig.savefig(os.path.join(save_fig_dir, savename + '.pdf'), format='pdf', bbox_inches='tight')
    plt.close(fig)


def get_top2_methods(col, is_largest):
    if is_largest:
        top2_name = col.nlargest(2).index.tolist()
    else:
        top2_name = col.nsmallest(2).index.tolist()
    name1, name2 = top2_name[0], top2_name[1]
    return name1, name2


def handle_one_col(df_metric, final_rate, is_largest):
    length = len(df_metric)
    res_start = int((1 - final_rate) * length)
    mean = df_metric[res_start:].mean()
    std = df_metric[res_start:].std()

    # mean.nlargest(2).index[1]
    res_latex = pd.Series(map(lambda mean, std: f"${mean:.3f}\pm {std:.3f}$", mean, std),
                          index=mean.index)
    res_excel = pd.Series(map(lambda mean, std: f"{mean:.3f}+{std:.3f}", mean, std),
                          index=mean.index)

    name1, name2 = get_top2_methods(mean, is_largest=is_largest)
    res_latex.loc[name1] = r"$\mathbf{" + r"{}".format(res_latex.loc[name1][1:-1]) + r"}$"
    res_latex.loc[name2] = r"\underline{" + res_latex.loc[name2] + r"}"

    return res_latex, res_excel


def handle_table(df_all, save_fig_dir, savename="all_results", final_rate=1):
    df_all.rename(columns={"FB": "Standard", "NX_0_": r"No Overlapping", "NX_10_": r"No Overlapping for 10 turns"},
                  level=0, inplace=True)
    df_all.rename(columns={"ifeat_feat": "MCD", "CV_turn": r"$\text{CV}_\text{M}$", "len_tra": "Length"}, level=1,
                  inplace=True)

    ways = df_all.columns.levels[0][::-1]
    metrics = df_all.columns.levels[1]
    methods = df_all.columns.levels[2].to_list()
    methods.remove("CIRS")
    methods.remove("CIRS w/o CI")
    methods = methods + ["CIRS", "CIRS w/o CI"]
    methods_order = dict(zip(methods, list(range(len(methods)))))

    df_latex = pd.DataFrame(columns=pd.MultiIndex.from_product([ways, metrics], names=["ways", "metrics"]))
    df_excel = pd.DataFrame(columns=pd.MultiIndex.from_product([ways, metrics], names=["ways", "metrics"]))

    for col, way in enumerate(ways):
        df = df_all[way]
        for row, metric in enumerate(metrics):
            df_metric = df[metric]
            is_largest = False if metric == "MCD" else True
            df_latex[way, metric], df_excel[way, metric] = handle_one_col(df_metric, final_rate, is_largest=is_largest)

    df_latex.sort_index(key=lambda index: [methods_order[x] for x in index.to_list()], inplace=True)
    df_excel.sort_index(key=lambda index: [methods_order[x] for x in index.to_list()], inplace=True)

    # print(df_latex.to_markdown())
    # excel_path = os.path.join(save_fig_dir, savename + '.xlsx')
    # df_excel.to_excel(excel_path)

    return df_latex, df_excel


def visual_one_group():
    realpath = os.path.dirname(__file__)
    save_fig_dir = os.path.join(realpath, "figures")

    create_dirs = [save_fig_dir]
    create_dir(create_dirs)

    dirpath = "./results_all_methods/kuaishou_len100"
    filenames = walk_paths(dirpath)
    dfs = loaddata(dirpath, filenames)

    ways = {'FB', 'NX_0_'}
    metrics = {'len_tra', 'CV', 'CV_turn', 'ifeat_feat'}
    df_all = organize_df(dfs, ways, metrics)

    print("Producing the table...")
    savename = "all_results"
    df_latex, df_excel = handle_table(df_all, save_fig_dir, savename=savename)

    # display(df_excel)
    # display(HTML(df_excel.to_html()))

    # please install openpyxl if you want to write to an excel file.
    # excel_path = os.path.join(save_fig_dir, savename + '.xlsx')
    # df_excel.to_excel(excel_path)


if __name__ == '__main__':
    visual_one_group()
