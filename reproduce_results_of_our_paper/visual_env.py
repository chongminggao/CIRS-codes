# -*- coding: utf-8 -*-
# @Author  : Chongming GAO
# @FileName: visual_env.py


import os

import re


import seaborn as sns
import pandas as pd

from matplotlib import pyplot as plt

from visualization.visual_utils import walk_paths, loaddata, organize_df

DATAPATH = "../environments/KuaishouRec/data"


def handle_data(df_all):
    data_r = df_all["R_tra"]
    meandata = data_r.mean()

    pattern_name = re.compile("(.*)[-\s]leave[=]?(\d)")

    df1 = pd.DataFrame()
    for k, v in meandata.iteritems():
        print(k)
        res = re.search(pattern_name, k)
        method = res.group(1)
        leave = res.group(2)
        df1.loc[leave, method] = v

    # df = df.columns[2:]
    # def mysub(name):
    #     return re.sub(r'\sw[:_]o\s', ' w/o ', name, flags=re.IGNORECASE)
    #
    # df1.columns = list(map(mysub, df1.columns))
    df1 = df1[sorted(df1.columns)]

    return df1



def visual_env(df_kuaishou, df_taobao, save_fig_dir, savename):
    # ways_kuaishou = df_kuaishou.columns.levels[0]
    # metrics_kuaishou = df_kuaishou.columns.levels[1]
    # methods_kuaishou = df_kuaishou.columns.levels[1]

    # ways_taobao = df_taobao.columns.levels[0]
    # metrics_taobao = df_taobao.columns.levels[1]
    # methods_taobao = df_taobao.columns.levels[1]



    all_method = sorted(set(df_kuaishou.columns.to_list() + df_taobao.columns.to_list()))
    color = sns.color_palette(n_colors=len(all_method))
    markers = ["o", "s", "p", "P", "X", "*", "h", "D", "v", "^", ">", "<"]

    color_kv = dict(zip(all_method, color))
    marker_kv = dict(zip(all_method, markers))

    color1 = [color_kv[k] for k in df_kuaishou.columns]
    color2 = [color_kv[k] for k in df_taobao.columns]
    marker1 = [marker_kv[k] for k in df_kuaishou.columns]
    marker2 = [marker_kv[k] for k in df_taobao.columns]


    fig = plt.figure(figsize=(5.5, 2))
    plt.subplots_adjust(wspace=0.4)

    ax2 = plt.subplot(121)
    # ax1 = plt.gca()
    ax2.set_ylabel("Accumulated reward", fontsize=11)
    ax2.set_xlabel(r"Distance threshold $d_Q$", fontsize=11)
    df_taobao.plot(kind="line", linewidth=1.8, ax=ax2, legend=None, color=color2, fillstyle='none', alpha=0.7, markeredgewidth=1.8)
    for i, line in enumerate(ax2.get_lines()):
        line.set_marker(marker2[i])
    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)
    plt.grid(linestyle='dashdot', linewidth=0.8)
    plt.xticks(range(0, 5), ["1.0", "2.0", "3.0", "4.0", "5.0"])


    ax1 = plt.subplot(122)
    # ax1 = plt.gca()
    ax1.set_ylabel("Accumulated reward", fontsize=11)
    ax1.set_xlabel(r"Window size $N$", fontsize=11)
    df_kuaishou.plot(kind="line", linewidth=1.8, ax=ax1, legend=None, color=color1, fillstyle='none', alpha=0.7, markeredgewidth=1.8)
    for i, line in enumerate(ax1.get_lines()):
        line.set_marker(marker1[i])
    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)
    plt.grid(linestyle='dashdot', linewidth=0.8)
    plt.xticks(range(0, 5), range(1, 6))
    ax1.set_title("KuaishouEnv", fontsize=11, y=1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    dict_label = dict(zip(labels1,lines1))
    dict_label.update(dict(zip(labels2,lines2)))
    dict_label1 = {r'$\epsilon$-greedy' if k == 'Epsilon-greedy' else k: v for k, v in dict_label.items()}
    ax2.legend(handles=dict_label1.values(), labels=dict_label1.keys(), ncol=5,
               loc='lower left', columnspacing=0.7,
               bbox_to_anchor=(-0.20, 1.13), fontsize=8.5)
    ax2.set_title("VirtualTaobao", fontsize=11, y=1)

    fig.savefig(os.path.join(save_fig_dir, savename + '.pdf'), format='pdf', bbox_inches='tight')
    plt.close(fig)

def main():
    realpath = os.path.dirname(__file__)
    save_fig_dir = os.path.join(realpath, "figures")

    dir1 = os.path.join(".", "results_leave", "kuaishou")
    filenames1 = walk_paths(dir1)
    df1 = loaddata(dir1, filenames1)

    dir2 = os.path.join(".", "results_leave", "taobao")
    filenames2 = walk_paths(dir2)
    df2 = loaddata(dir2, filenames2)

    # df_kuaishou = handle_data(df1, rl_threshold=100)
    # df_taobao = handle_data(df2, rl_threshold=0)

    ways={'FB', 'NX_0_', 'NX_10_'}
    metrics={'ctr', 'len_tra', 'R_tra',  'CV', 'CV_turn', 'ifeat_feat'}
    df_kuaishou = organize_df(df1, ways, metrics)

    ways = {'FB'}
    metrics = {'ctr', 'len_tra', 'R_tra'}
    df_taobao = organize_df(df2, ways, metrics)


    df_ks = df_kuaishou['FB']
    df_tb = df_taobao['FB']

    df_ks = handle_data(df_ks)
    df_tb = handle_data(df_tb)

    visual_env(df_ks, df_tb, save_fig_dir, savename="leave")


if __name__ == '__main__':
    # args = get_args()
    main()
