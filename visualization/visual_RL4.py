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

def get_args():
    parser = argparse.ArgumentParser()
    # --result_dir "./visualization/results/KuaishouEnv-v0"
    parser.add_argument("--result_dir", type=str, default="./saved_models/VirtualTB-v0/CIRS/logs")
    parser.add_argument("--use_filename", type=str, default="Yes")
    # parser.add_argument("--result_dir", type=str, default="../saved_models/PPO_realEnv/logs")

    args = parser.parse_known_args()[0]
    return args


def walk_paths(result_dir):
    g = os.walk(result_dir)

    files = []
    for path, dir_list, file_list in g:
        for file_name in file_list:
            if file_name[0] == '.' or file_name[0] == '_':
                continue
            print(os.path.join(path, file_name))
            files.append(file_name)
    return files


def loaddata(dirpath, filenames, args, is_info=False):
    pattern_epoch = re.compile("Epoch: \[(\d+)]")
    pattern_info = re.compile("Info: \[(\{.+\})]")
    pattern_message = re.compile('"message": "(.+)"')
    pattern_array = re.compile("array\((.+?)\)")

    pattern_tau = re.compile('"tau": (.+),')
    pattern_read = re.compile('"read_message": "(.+)"')

    dfs = {}
    infos = {}
    df = pd.DataFrame()
    for filename in filenames:
        # if filename == ".DS_Store":
        #     continue
        if filename[0] == '.' or filename[0] == '_':
            continue
        df = pd.DataFrame()
        message = "None"
        filepath = os.path.join(dirpath, filename)
        with open(filepath, "r") as file:
            lines = file.readlines()
            start = False
            add = 0
            info_extra = {'tau':0, 'read':""}
            for i, line in enumerate(lines):
                res_tau = re.search(pattern_tau, line)
                if res_tau:
                    info_extra['tau'] = res_tau.group(1)
                res_read = re.search(pattern_read, line)
                if res_read:
                    info_extra['read'] = res_read.group(1)

                res = re.search(pattern_epoch, line)
                if res:
                    epoch = int(res.group(1))
                    if (start == False) and epoch == 0:
                        add = 1
                        start = True
                    epoch += add
                    info = re.search(pattern_info, line)
                    try:
                        info1 = info.group(1).replace("\'", "\"")
                    except Exception as e:
                        print("jump incomplete line: [{}]".format(line))
                        continue
                    info2 = re.sub(pattern_array, lambda x: x.group(1), info1)

                    data = json.loads(info2)
                    df_data = pd.DataFrame(data, index=[epoch],dtype=float)
                    # df = df.append(df_data)
                    df = pd.concat([df, df_data])
                res_message = re.search(pattern_message, line)
                if res_message:
                    message = res_message.group(1)

            if args.use_filename == "Yes":
                message = filename[:-4]

            # print(file.name)
            df.rename(
                columns={"RL_val_trajectory_reward": "R_tra",
                         "RL_val_trajectory_len": 'len_tra',
                         "RL_val_CTR": 'ctr'},
                inplace=True)
            # print("JJ", filename)
            df = df[["R_tra","len_tra","ctr"]]

        dfs[message] = df
        infos[message] = info_extra

    dfs = OrderedDict(sorted(dfs.items(), key=lambda item: len(item[1]), reverse=True))

    indices = [list(dfs.keys()), df.columns.to_list()]

    df_all = pd.DataFrame(columns=pd.MultiIndex.from_product(indices, names=["Exp", "metrics"]))

    for message, df in dfs.items():
        # print(message, df)
        for col in df.columns:
            df_all[message, col] = df[col]

    # # Rename MultiIndex columns in Pandas
    # # https://stackoverflow.com/questions/41221079/rename-multiindex-columns-in-pandas
    # df_all.rename(
    #     columns={"RL_val_trajectory_reward": "R_tra", "RL_val_trajectory_len": 'len_tra', "RL_val_CTR": 'ctr'},
    #     level=1,inplace=True)

    # change order of levels
    # https://stackoverflow.com/questions/29859296/how-do-i-change-order-grouping-level-of-pandas-multiindex-columns
    df_all.columns = df_all.columns.swaplevel(0, 1)
    df_all.sort_index(axis=1, level=0, inplace=True)

    if is_info:
        return df_all, infos

    return df_all

def axis_shift(ax1 ,x_shift=0.01, y_shift=0):
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

    pattern="\[([KT]_)?(.+)\]"
    all_method_map = {}
    for method in all_method:
        res = re.match("\[([KT]_)?(.+?)(_len.+)?\]", method)
        if res:
            all_method_map[method] = res.group(2)
    pprint.pprint(all_method_map)

    methods_list = list(set(all_method_map.values()))
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
        alpha=series[index]
        cnt = 1
        df = dfs[index]

        new_name = {name:all_method_map[name] for name in df.columns.levels[1]}
        df = df.rename(columns=new_name, level=1)

        data_r = df[visual_cols[0]]
        data_len = df[visual_cols[1]]
        data_ctr = df[visual_cols[2]]

        color = [color_kv[name] for name in data_r.columns]
        marker = [marker_kv[name] for name in data_r.columns]

        ax1 = plt.subplot2grid((3,4), (0,index))
        data_r.plot(kind="line", linewidth=1, ax=ax1, legend=None, color=color, markevery=int(len(data_r)/10), fillstyle='none', alpha=.8, markersize=3)
        for i, line in enumerate(ax1.get_lines()):
            line.set_marker(marker[i])
        plt.yticks(fontsize=10)
        plt.xticks(fontsize=10)
        plt.grid(linestyle='dashdot', linewidth=0.8)
        # ax1.set_title("({}{})".format(alpha, cnt), fontsize=fontsize, y=.97)
        ax1.set_title("({}{})".format(alpha, cnt), fontsize=fontsize, loc="left", x=0.4, y=.97)
        ax1.set_title("{}".format(dataset[index]), fontsize=fontsize, y=1.1, fontweight=700)
        # ax1.set_title("{}".format(dataset[index]), fontsize=fontsize, loc="left", x=0.2, y=1.1, fontweight=700)
        # ax1.set_xlabel("({})".format(series[cnt]), fontsize=fontsize)
        cnt += 1
        # ax1.set_xticklabels(["{:.0%}".format(i) for i in ax1.get_xticks()])
        # plt.xticks([0, 25, 50], [0, 25, 50], rotation=0)
        # ax1.xaxis.set_label_coords(0.5, -0.2)

        ax2 = plt.subplot2grid((3,4), (1,index))
        data_len.plot(kind="line", linewidth=1, ax=ax2, legend=None, color=color, markevery=int(len(data_r)/10), fillstyle='none', alpha=.8, markersize=3)
        for i, line in enumerate(ax2.get_lines()):
            line.set_marker(marker[i])
        plt.yticks(fontsize=10)
        plt.xticks(fontsize=10)
        plt.grid(linestyle='dashdot', linewidth=0.8)
        ax2.set_title("({}{})".format(alpha, cnt), fontsize=fontsize, y=.97)
        ax2.set_title(r"$\it{Max round=" + str(maxlen[index]) + r"}$", fontsize=fontsize-1.5, loc="left", x=-0.2, y=.97)
        # ax2.set_xlabel("({})".format(series[cnt]), fontsize=fontsize)
        cnt += 1
        # ax1.set_xticklabels(["{:.0%}".format(i) for i in ax1.get_xticks()])
        # plt.xticks([0, 25, 50], [0, 25, 50], rotation=0)
        # ax1.xaxis.set_label_coords(0.5, -0.2)

        ax3 = plt.subplot2grid((3,4), (2,index))
        data_ctr.plot(kind="line", linewidth=1, ax=ax3, legend=None, color=color, markevery=int(len(data_r)/10), fillstyle='none', alpha=.8, markersize=3)
        for i, line in enumerate(ax3.get_lines()):
            line.set_marker(marker[i])
        plt.yticks(fontsize=10)
        plt.xticks(fontsize=10)
        ax3.set_title("({}{})".format(alpha, cnt), fontsize=fontsize, y=.97)
        # ax3.set_xlabel("({})\n".format(series[cnt]) + r'$\bf{epoch}$',
        #                fontsize=fontsize)
        ax3.set_xlabel("epoch", fontsize=11)
        cnt += 1
        plt.grid(linestyle='dashdot', linewidth=0.8)
        # ax1.set_xticklabels(["{:.0%}".format(i) for i in ax1.get_xticks()])
        # plt.xticks([0, 25, 50], [0, 25, 50], rotation=0)

        # ax1.xaxis.set_label_coords(0.5, -0.2)
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
    # def mysub(name):
    #     return re.sub(r'\sw[:_]o\s', ' w/o ', name, flags=re.IGNORECASE)
    # legend_name = list(map(mysub, legend_name))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax4.get_legend_handles_labels()
    dict_label = dict(zip(labels1, lines1))
    dict_label.update(dict(zip(labels2, lines2)))
    dict_label = OrderedDict(sorted(dict_label.items(), key=lambda x: x[0]))

    dict_label = {'CIRS w/o CI' if k=='CIRS w_o CI' or k=='CIRSwoCI' else k :v for k,v in dict_label.items()}
    dict_label = {r'$\epsilon$-greedy' if k == 'Epsilon Greedy' or k == 'epsilon-greedy' else k: v for k, v in dict_label.items()}
    dict_label = {r'DeepFM' if k == 'DeepFM+Softmax'else k: v for k, v in dict_label.items()}

    ax1.legend(handles=dict_label.values(), labels=dict_label.keys(), ncol=10,
               loc='lower left', columnspacing=0.7,
               bbox_to_anchor=(-0.20, 1.24), fontsize=10.5)

    axo = plt.axes([0, 0, 1, 1], facecolor=(1, 1, 1, 0))
    x, y = np.array([[0.505, 0.505], [0.06, 0.92]])
    line = Line2D(x, y, lw=3, linestyle="dotted", color=(0.5,0.5,0.5))
    axo.add_line(line)
    plt.text(0.16, 0.02, "(A-B) Results with large interaction rounds", fontsize=11, fontweight=400)
    plt.text(0.58, 0.02, "(C-D) Results with limited interaction rounds", fontsize=11, fontweight=400)
    # plt.axis('off')

    fig.savefig(os.path.join(save_fig_dir, savename + '.pdf'), format='pdf', bbox_inches='tight')
    plt.close(fig)
    a = 1





def main(args):


    realpath = os.path.dirname(__file__)
    save_fig_dir = os.path.join(realpath, "figures")

    create_dirs = [save_fig_dir]
    create_dir(create_dirs)

    dirpath = "./results"

    result_dir1 = os.path.join(dirpath, "taobao_len50")
    filenames = walk_paths(result_dir1)
    df1 = loaddata(result_dir1, filenames, args)

    result_dir2 = os.path.join(dirpath, "kuaishou_len100")
    filenames = walk_paths(result_dir2)
    df2 = loaddata(result_dir2, filenames, args)

    result_dir3 = os.path.join(dirpath, "taobao_len10")
    filenames = walk_paths(result_dir3)
    df3 = loaddata(result_dir3, filenames, args)

    result_dir4 = os.path.join(dirpath, "kuaishou_len30")
    filenames = walk_paths(result_dir4)
    df4 = loaddata(result_dir4, filenames, args)

    visual4(df1, df2, df3, df4, save_fig_dir, savename="main_result")


if __name__ == '__main__':
    args = get_args()
    main(args)
