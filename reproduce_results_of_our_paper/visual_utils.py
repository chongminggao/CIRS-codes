# -*- coding: utf-8 -*-
# @Time    : 2022/12/23 23:57
# @Author  : Chongming GAO
# @FileName: visual_utils.py

import os
import json
from collections import OrderedDict
import pandas as pd
import re


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


def organize_df(dfs, ways, metrics):
    indices = [list(dfs.keys()), ways, metrics]

    df_all = pd.DataFrame(columns=pd.MultiIndex.from_product(indices, names=["Exp", "ways", "metrics"]))

    for message, df in dfs.items():
        for way in ways:
            for metric in metrics:
                col = (way if way != "FB" else "") + metric
                df_all[message, way, metric] = df[col]

    # # Rename MultiIndex columns in Pandas
    # # https://stackoverflow.com/questions/41221079/rename-multiindex-columns-in-pandas
    # df_all.rename(
    #     columns={"RL_val_trajectory_reward": "R_tra", "RL_val_trajectory_len": 'len_tra', "RL_val_CTR": 'ctr'},
    #     level=1,inplace=True)

    # change order of levels
    # https://stackoverflow.com/questions/29859296/how-do-i-change-order-grouping-level-of-pandas-multiindex-columns

    df_all.columns = df_all.columns.swaplevel(0, 2)
    df_all.sort_index(axis=1, level=0, inplace=True)
    df_all.columns = df_all.columns.swaplevel(0, 1)

    all_method = set(df_all.columns.levels[2].to_list())
    all_method_map = {}
    for method in all_method:
        res = re.match("\[([KT]_)?(.+?)(_len.+)?\]", method)
        if res:
            all_method_map[method] = res.group(2)

    df_all.rename(
        columns=all_method_map,
        level=2, inplace=True)

    df_all.rename(
        columns={"CIRSwoCI": 'CIRS w/o CI',
                 "epsilon-greedy": r'$\epsilon$-greedy',
                 "DeepFM+Softmax": 'DeepFM'},
        level=2, inplace=True)

    return df_all


def loaddata(dirpath, filenames, use_filename=True):
    pattern_epoch = re.compile("Epoch: \[(\d+)]")
    pattern_info = re.compile("Info: \[(\{.+\})]")
    pattern_message = re.compile('"message": "(.+)"')
    pattern_array = re.compile("array\((.+?)\)")

    pattern_tau = re.compile('"tau": (.+),')
    pattern_read = re.compile('"read_message": "(.+)"')

    dfs = {}
    infos = {}

    for filename in filenames:
        if filename[0] == '.' or filename[0] == '_':  # ".DS_Store":
            continue
        df = pd.DataFrame()
        message = "None"
        filepath = os.path.join(dirpath, filename)
        with open(filepath, "r") as file:
            lines = file.readlines()
            start = False
            add = 0
            info_extra = {'tau': 0, 'read': ""}
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
                    df_data = pd.DataFrame(data, index=[epoch], dtype=float)
                    # df = df.append(df_data)
                    df = pd.concat([df, df_data])
                res_message = re.search(pattern_message, line)
                if res_message:
                    message = res_message.group(1)

            if use_filename:
                message = filename[:-4]

            # print(file.name)
            df.rename(
                columns={"RL_val_trajectory_reward": "R_tra",
                         "RL_val_trajectory_len": 'len_tra',
                         "RL_val_CTR": 'ctr'},
                inplace=True)

            df.rename(
                columns={"trajectory_reward": "R_tra",
                         "trajectory_len": 'len_tra',
                         "CTR": 'ctr'},
                inplace=True)

        dfs[message] = df
        infos[message] = info_extra

    dfs = OrderedDict(sorted(dfs.items(), key=lambda item: len(item[1]), reverse=True))
    return dfs
