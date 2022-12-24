# -*- coding: utf-8 -*-
# @Time    : 2021/9/13 10:25 上午
# @Author  : Chongming GAO
# @FileName: visual_RL.py


import os

from collections import OrderedDict

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

from util.utils import create_dir
import seaborn as sns

from visualization.visual_utils import walk_paths, organize_df, loaddata


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


def visual4(df1, df2, df3, df4, save_fig_dir, savename="three"):
    visual_cols = ['R_tra', 'len_tra', 'ctr']

    df1 = df1.iloc[:100]
    df2 = df2.iloc[:200]
    df3 = df3.iloc[:200]
    df4 = df4.iloc[:1000]

    # compute_improvement(df1, col="R_tra", last=0)
    # compute_improvement(df2, col="R_tra", last=0)
    # compute_improvement(df3, col="R_tra", last=0)
    # compute_improvement(df4, col="R_tra", last=0)
    #
    # compute_improvement(df1, col="R_tra", last=10)
    # compute_improvement(df2, col="R_tra", last=10)
    # compute_improvement(df3, col="R_tra", last=10)
    # compute_improvement(df4, col="R_tra", last=10)
    #
    # compute_improvement(df1, col="ctr", last=10)
    # compute_improvement(df2, col="ctr", last=10)
    # compute_improvement(df3, col="ctr", last=10)
    # compute_improvement(df4, col="ctr", last=10)
    #
    # compute_improvement(df1, col="len_tra", last=10)
    # compute_improvement(df2, col="len_tra", last=10)
    # compute_improvement(df3, col="len_tra", last=10)
    # compute_improvement(df4, col="len_tra", last=10)

    dfs = [df1, df2, df3, df4]
    series = "ABCD"
    dataset = ["VirtualTaobao", "KuaiEnv", "VirtualTaobao", "KuaiEnv"]
    maxlen = [50, 100, 10, 30]
    fontsize = 11.5

    all_method = sorted(set(df1['R_tra'].columns.to_list() +
                            df2['R_tra'].columns.to_list() +
                            df3['R_tra'].columns.to_list() +
                            df4['R_tra'].columns.to_list()))


    methods_list = list(all_method)

    num_methods = len(methods_list)

    colors = sns.color_palette(n_colors=num_methods)
    markers = ["o", "s", "p", "P", "X", "*", "h", "D", "v", "^", ">", "<", "x", "H"][:num_methods]

    color_kv = dict(zip(methods_list, colors))
    marker_kv = dict(zip(methods_list, markers))

    fig = plt.figure(figsize=(12, 7))
    # plt.subplots_adjust(wspace=0.3)
    plt.subplots_adjust(wspace=0.3)
    plt.subplots_adjust(hspace=0.3)
    axs = []
    for index in range(len(dfs)):
        alpha = series[index]
        cnt = 1
        df = dfs[index]

        data_r = df[visual_cols[0]]
        data_len = df[visual_cols[1]]
        data_ctr = df[visual_cols[2]]

        color = [color_kv[name] for name in data_r.columns]
        marker = [marker_kv[name] for name in data_r.columns]

        ax1 = plt.subplot2grid((3, 4), (0, index))
        data_r.plot(kind="line", linewidth=1, ax=ax1, legend=None, color=color, markevery=int(len(data_r) / 10),
                    fillstyle='none', alpha=.8, markersize=3)
        for i, line in enumerate(ax1.get_lines()):
            line.set_marker(marker[i])
        plt.yticks(fontsize=10)
        plt.xticks(fontsize=10)
        plt.grid(linestyle='dashdot', linewidth=0.8)
        ax1.set_title("({}{})".format(alpha, cnt), fontsize=fontsize, loc="left", x=0.4, y=.97)
        ax1.set_title("{}".format(dataset[index]), fontsize=fontsize, y=1.1, fontweight=700)
        cnt += 1

        ax2 = plt.subplot2grid((3, 4), (1, index))
        data_len.plot(kind="line", linewidth=1, ax=ax2, legend=None, color=color, markevery=int(len(data_r) / 10),
                      fillstyle='none', alpha=.8, markersize=3)
        for i, line in enumerate(ax2.get_lines()):
            line.set_marker(marker[i])
        plt.yticks(fontsize=10)
        plt.xticks(fontsize=10)
        plt.grid(linestyle='dashdot', linewidth=0.8)
        ax2.set_title("({}{})".format(alpha, cnt), fontsize=fontsize, y=.97)
        ax2.set_title(r"$\it{Max round=" + str(maxlen[index]) + r"}$", fontsize=fontsize - 1.5, loc="left", x=-0.2,
                      y=.97)
        cnt += 1

        ax3 = plt.subplot2grid((3, 4), (2, index))
        data_ctr.plot(kind="line", linewidth=1, ax=ax3, legend=None, color=color, markevery=int(len(data_r) / 10),
                      fillstyle='none', alpha=.8, markersize=3)
        for i, line in enumerate(ax3.get_lines()):
            line.set_marker(marker[i])
        plt.yticks(fontsize=10)
        plt.xticks(fontsize=10)
        ax3.set_title("({}{})".format(alpha, cnt), fontsize=fontsize, y=.97)
        ax3.set_xlabel("epoch", fontsize=11)
        cnt += 1
        plt.grid(linestyle='dashdot', linewidth=0.8)
        if index == 2:
            axis_shift(ax1, .015)
            axis_shift(ax2, .015)
            axis_shift(ax3, .015)
        if index == 3:
            axis_shift(ax1, .005)
            axis_shift(ax2, .005)
            axis_shift(ax3, .005)
        axs.append((ax1, ax2, ax3))

    ax1, ax2, ax3 = axs[0]
    ax1.set_ylabel("Cumulative satisfaction", fontsize=10, fontweight=700)
    ax2.set_ylabel("Interaction length", fontsize=10, fontweight=700)
    ax3.set_ylabel("Single-round satisfaction", fontsize=10, fontweight=700)
    ax3.yaxis.set_label_coords(-0.17, 0.5)

    ax4 = axs[1][0]

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax4.get_legend_handles_labels()
    dict_label = dict(zip(labels1, lines1))
    dict_label.update(dict(zip(labels2, lines2)))
    dict_label = OrderedDict(sorted(dict_label.items(), key=lambda x: x[0]))

    dict_label = {'CIRS w/o CI' if k == 'CIRS w_o CI' or k == 'CIRSwoCI' else k: v for k, v in dict_label.items()}
    dict_label = {r'$\epsilon$-greedy' if k == 'Epsilon Greedy' or k == 'epsilon-greedy' else k: v for k, v in
                  dict_label.items()}
    dict_label = {r'DeepFM' if k == 'DeepFM+Softmax' else k: v for k, v in dict_label.items()}

    ax1.legend(handles=dict_label.values(), labels=dict_label.keys(), ncol=10,
               loc='lower left', columnspacing=0.7,
               bbox_to_anchor=(-0.20, 1.24), fontsize=10.5)

    axo = plt.axes([0, 0, 1, 1], facecolor=(1, 1, 1, 0))
    x, y = np.array([[0.505, 0.505], [0.06, 0.92]])
    line = Line2D(x, y, lw=3, linestyle="dotted", color=(0.5, 0.5, 0.5))
    axo.add_line(line)
    plt.text(0.16, 0.02, "(A-B) Results with large interaction rounds", fontsize=11, fontweight=400)
    plt.text(0.58, 0.02, "(C-D) Results with limited interaction rounds", fontsize=11, fontweight=400)
    # plt.axis('off')

    fig.savefig(os.path.join(save_fig_dir, savename + '.pdf'), format='pdf', bbox_inches='tight')
    plt.close(fig)


def visual_four_groups():
    realpath = os.path.dirname(__file__)
    save_fig_dir = os.path.join(realpath, "figures")

    create_dirs = [save_fig_dir]
    create_dir(create_dirs)

    dirpath = "./results_all_methods"

    ways = {'FB'}
    metrics = {'ctr', 'len_tra', 'R_tra'}
    result_dir1 = os.path.join(dirpath, "taobao_len50")
    filenames = walk_paths(result_dir1)
    dfs1 = loaddata(result_dir1, filenames)
    df1 = organize_df(dfs1, ways, metrics)

    ways = {'FB'}
    metrics = {'ctr', 'len_tra', 'R_tra'}
    result_dir3 = os.path.join(dirpath, "taobao_len10")
    filenames = walk_paths(result_dir3)
    dfs3 = loaddata(result_dir3, filenames)
    df3 = organize_df(dfs3, ways, metrics)

    ways = {'FB', 'NX_0_', 'NX_10_'}
    metrics = {'ctr', 'len_tra', 'R_tra', 'CV', 'CV_turn', 'ifeat_feat'}
    result_dir2 = os.path.join(dirpath, "kuaishou_len100")
    filenames = walk_paths(result_dir2)
    dfs2 = loaddata(result_dir2, filenames)
    df2 = organize_df(dfs2, ways, metrics)

    ways = {'FB', 'NX_0_', 'NX_10_'}
    metrics = {'ctr', 'len_tra', 'R_tra', 'CV', 'CV_turn', 'ifeat_feat'}
    result_dir4 = os.path.join(dirpath, "kuaishou_len30")
    filenames = walk_paths(result_dir4)
    dfs4 = loaddata(result_dir4, filenames)
    df4 = organize_df(dfs4, ways, metrics)

    way = "FB"
    df1, df2, df3, df4 = df1[way], df2[way], df3[way], df4[way]
    visual4(df1, df2, df3, df4, save_fig_dir, savename="main_result")


if __name__ == '__main__':
    visual_four_groups()
