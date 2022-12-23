# -*- coding: utf-8 -*-
# @Time    : 2021/9/13 10:25 上午
# @Author  : Chongming GAO
# @FileName: visual_RL.py

import argparse
import os
import re
import json
from collections import OrderedDict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

from util.utils import create_dir
import seaborn as sns
import re
import pprint

from visualization.visual_utils import walk_paths, loaddata, organize_df


# def get_args():
#     parser = argparse.ArgumentParser()
#     # --result_dir "./results/kuaishou_len30"
#     parser.add_argument("--result_dir", type=str, default="./results_all_methods/kuaishou_len100")
#     parser.add_argument("--use_filename", type=str, default="Yes")
#     # parser.add_argument("--result_dir", type=str, default="../saved_models/PPO_realEnv/logs")
#
#     args = parser.parse_known_args()[0]
#     return args

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
    # ax1.set_title("({}{})".format(alpha, cnt), fontsize=fontsize, y=.97)
    # ax1.set_title("({}{})".format(alpha, cnt), fontsize=fontsize, loc="left", x=0.4, y=.97)
    # ax1.set_title("{}".format(dataset[index]), fontsize=fontsize, y=1.1, fontweight=700)


def visual(df_all, save_fig_dir, savename="three"):
    # visual_cols = ['R_tra', 'len_tra', 'ctr']

    ways = df_all.columns.levels[0]
    metrics = df_all.columns.levels[1]
    methods = df_all.columns.levels[2]


    # all_method = set(df_all.columns.levels[2].to_list())
    # all_method_map = {}
    # for method in all_method:
    #     res = re.match("\[([KT]_)?(.+?)(_len.+)?\]", method)
    #     if res:
    #         all_method_map[method] = res.group(2)
    # pprint.pprint(all_method_map)
    #
    # new_name = {name: all_method_map[name] for name in df_all.columns.levels[2]}
    # df_all = df_all.rename(columns=new_name, level=2)

    # series = "ABCD"
    # dataset = ["VirtualTaobao", "KuaiEnv", "VirtualTaobao", "KuaiEnv"]
    # maxlen = [50, 100, 10, 30]
    fontsize = 11.5

    methods_list = list(set(methods))
    num_methods = len(methods_list)

    colors = sns.color_palette(n_colors=num_methods)
    markers = ["o", "s", "p", "P", "X", "*", "h", "D", "v", "^", ">", "<", "x", "H"][:num_methods]

    color_kv = dict(zip(methods_list, colors))
    marker_kv = dict(zip(methods_list, markers))

    fig = plt.figure(figsize=(10, 15))
    plt.subplots_adjust(wspace=0.3)
    plt.subplots_adjust(hspace=0.3)

    ways = df_all.columns.levels[0]
    metrics = df_all.columns.levels[1]
    axs = np.empty([len(metrics), len(ways)], dtype=object)
    for col, way in enumerate(ways):
        df = df_all[way]

        CV = df[metrics[0]]
        CV_turn = df[metrics[1]]
        data_r = df[metrics[2]]
        data_len = df[metrics[3]]
        data_ctr = df[metrics[4]]

        color = [color_kv[name] for name in data_r.columns]
        marker = [marker_kv[name] for name in data_r.columns]

        for row, metric in enumerate(metrics):
            print(metric, row, col)
            df_metric = df[metric]
            ax1 = plt.subplot2grid((len(metrics), len(ways)), (row, col))
            axs[row, col] = ax1
            draw(df_metric, ax1, color, marker, metric)

    ax_legend = axs[0][1]
    lines, labels = ax_legend.get_legend_handles_labels()
    dict_label = dict(zip(labels, lines))
    dict_label = OrderedDict(sorted(dict_label.items(), key=lambda x: x[0]))

    # dict_label = {'CIRS w/o CI' if k == 'CIRS w_o CI' or k == 'CIRSwoCI' else k: v for k, v in dict_label.items()}
    # dict_label = {r'$\epsilon$-greedy' if k == 'Epsilon Greedy' or k == 'epsilon-greedy' else k: v for k, v in
    #               dict_label.items()}
    # dict_label = {r'DeepFM' if k == 'DeepFM+Softmax' else k: v for k, v in dict_label.items()}

    ax_legend.legend(handles=dict_label.values(), labels=dict_label.keys(), ncol=10,
                     loc='lower left', columnspacing=0.7,
                     bbox_to_anchor=(-0.20, 1.24), fontsize=10.5)

    # lines1, labels1 = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax4.get_legend_handles_labels()
    # dict_label = dict(zip(labels1, lines1))
    # dict_label.update(dict(zip(labels2, lines2)))
    # dict_label = OrderedDict(sorted(dict_label.items(), key=lambda x: x[0]))
    # dict_label = {'CIRS w/o CI' if k == 'CIRS w_o CI' or k == 'CIRSwoCI' else k: v for k, v in dict_label.items()}
    # dict_label = {r'$\epsilon$-greedy' if k == 'Epsilon Greedy' or k == 'epsilon-greedy' else k: v for k, v in
    #               dict_label.items()}
    # dict_label = {r'DeepFM' if k == 'DeepFM+Softmax' else k: v for k, v in dict_label.items()}
    # ax1.legend(handles=dict_label.values(), labels=dict_label.keys(), ncol=10,
    #            loc='lower left', columnspacing=0.7,
    #            bbox_to_anchor=(-0.20, 1.24), fontsize=10.5)
    #
    # axo = plt.axes([0, 0, 1, 1], facecolor=(1, 1, 1, 0))
    # x, y = np.array([[0.505, 0.505], [0.06, 0.92]])
    # line = Line2D(x, y, lw=3, linestyle="dotted", color=(0.5, 0.5, 0.5))
    # axo.add_line(line)
    # plt.text(0.16, 0.02, "(A-B) Results with large interaction rounds", fontsize=11, fontweight=400)
    # plt.text(0.58, 0.02, "(C-D) Results with limited interaction rounds", fontsize=11, fontweight=400)
    # plt.axis('off')

    fig.savefig(os.path.join(save_fig_dir, savename + '.pdf'), format='pdf', bbox_inches='tight')
    plt.close(fig)
    a = 1

def handle_table(df_all, save_fig_dir, savename="all_results"):
    ways = df_all.columns.levels[0]
    metrics = df_all.columns.levels[1]
    methods = df_all.columns.levels[2]

    df_latex = pd.DataFrame(columns=pd.MultiIndex.from_product([ways, metrics], names=["ways", "metrics"]))
    df_excel = pd.DataFrame(columns=pd.MultiIndex.from_product([ways, metrics], names=["ways", "metrics"]))
    def handle(df_metric):
        mean = df_metric.mean()
        std = df_metric.std()
        res_latex = pd.Series(map(lambda mean, std: f"${mean:.3f}\pm {std:.3f}$", mean, std),
                        index=mean.index)
        res_excel = pd.Series(map(lambda mean, std: f"{mean:.3f}+{std:.3f}", mean, std),
                              index=mean.index)
        return res_latex, res_excel

    for col, way in enumerate(ways):
        df = df_all[way]
        for row, metric in enumerate(metrics):
            # print(metric, row, col)
            df_metric = df[metric]
            df_latex[way, metric], df_excel[way, metric] = handle(df_metric)
            # df_excel

    print(df_latex.to_latex(escape=False))
    excel_path = os.path.join(save_fig_dir, savename + '.xlsx')

    df_excel.to_excel(excel_path)


def main():
    realpath = os.path.dirname(__file__)
    save_fig_dir = os.path.join(realpath, "figures")

    create_dirs = [save_fig_dir]
    create_dir(create_dirs)

    dirpath = "./results_all_methods/kuaishou_len100"

    filenames = walk_paths(dirpath)
    dfs = loaddata(dirpath, filenames)
    df_all = organize_df(dfs)


    handle_table(df_all, save_fig_dir, savename="all_results")
    visual(df_all, save_fig_dir, savename="all_results")


if __name__ == '__main__':
    # args = get_args()
    main()
