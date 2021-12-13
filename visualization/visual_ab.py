# -*- coding: utf-8 -*-
# @Time    : 2021/9/13 10:25 上午
# @Author  : Chongming GAO
# @FileName: visual_RL.py

import argparse
import collections
import os
import pickle
import re
import json
from collections import OrderedDict

import matplotlib
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
import matplotlib.gridspec as gridspec
import seaborn as sns

from visualization.seabornfig2grid import SeabornFig2Grid


from core.user_model_pairwise import UserModel_Pairwise
from util.utils import create_dir

DATAPATH = "../environments/KuaishouRec/data"
def get_args():
    parser = argparse.ArgumentParser()
    # --result_dir "./visualization/results/KuaishouEnv-v0"
    parser.add_argument("--user_model_name", type=str, default="DeepFM-pairwise")
    parser.add_argument("--env", type=str, default="KuaishouEnv-v0")
    parser.add_argument("--read_message", type=str, default="visual_ab")
    parser.add_argument('--is_ab', dest='is_ab', action='store_true')
    parser.add_argument('--no_ab', dest='is_ab', action='store_false')
    parser.set_defaults(is_ab=True)

    args = parser.parse_known_args()[0]
    return args


def walk_paths(result_dir):
    g = os.walk(result_dir)

    files = []
    for path, dir_list, file_list in g:
        for file_name in file_list:
            print(os.path.join(path, file_name))
            files.append(file_name)
    return files


def loaddata(args):

    USERMODEL_Path = os.path.join("..", "saved_models", args.env, args.user_model_name)
    model_parameter_path = os.path.join(USERMODEL_Path,
                                        "{}_params_{}.pickle".format(args.user_model_name, args.read_message))
    model_save_path = os.path.join(USERMODEL_Path, "{}_{}.pt".format(args.user_model_name, args.read_message))

    with open(model_parameter_path, "rb") as file:
        model_params = pickle.load(file)

    model_params["device"] = "cpu"
    user_model = UserModel_Pairwise(**model_params)
    user_model.load_state_dict(torch.load(model_save_path))

    if hasattr(user_model, 'ab_embedding_dict') and args.is_ab:
        alpha_u = user_model.ab_embedding_dict["alpha_u"].weight.detach().cpu().numpy()
        beta_i = user_model.ab_embedding_dict["beta_i"].weight.detach().cpu().numpy()
    else:
        print("Note there are no available alpha and beta！！")
        alpha_u = np.ones([7176, 1])
        beta_i = np.ones([10729, 1])

    filename = os.path.join(DATAPATH, "big_matrix.csv")
    df_big = pd.read_csv(filename, usecols=['user_id', 'photo_id', 'timestamp', 'watch_ratio', 'photo_duration'])



    return alpha_u, beta_i, df_big


def visual(alpha_u, beta_i, df_big, save_fig_dir, savename = "alpha_beta"):

    user_cnt = collections.Counter(df_big['user_id'])
    item_cnt = collections.Counter(df_big['photo_id'])

    lbe_user = LabelEncoder()
    lbe_user.fit(np.array(list(user_cnt.keys())))
    lbe_item = LabelEncoder()
    lbe_item.fit(np.array(list(item_cnt.keys())))

    # df_big['alpha_u'] = df_big['user_id'].map(lambda x: alpha_u[x])
    # df_big['beta_i'] = df_big['photo_id'].map(lambda x: beta_i[x])

    item_cnt[1225] = 0
    user_cnt = OrderedDict(sorted(user_cnt.items(), key=lambda x: x[0]))
    item_cnt = OrderedDict(sorted(item_cnt.items(), key=lambda x: x[0]))

    df_a = pd.DataFrame({"alpha": alpha_u.squeeze(), "popularity": user_cnt.values()})
    df_b = pd.DataFrame({"beta": beta_i.squeeze(), "popularity": item_cnt.values()})

    df_a = df_a[df_a["alpha"] < 0.06]
    df_b = df_b[df_b["beta"] <0.08]

    # sns.jointplot(x=alpha_u.squeeze(), y=np.array(list(user_cnt.values())), kind="kde")
    # g1.savefig(os.path.join(save_fig_dir, savename + '.pdf'), format='pdf')

    # plt.scatter(alpha_u, user_cnt.values())

    # plt.scatter(beta_i, item_cnt.values())

    # https://stackoverflow.com/questions/34706845/change-xticklabels-fontsize-of-seaborn-heatmap
    # https://seaborn.pydata.org/generated/seaborn.set_theme.html#seaborn.set_theme
    sns.set(style="ticks", font_scale=1.5)
    savename1="alpha_popularity"
    g1 = sns.jointplot(data=df_a, x="alpha", y="popularity",
                       marker="x", s=100, space=0, height=5)
    g1.plot_joint(sns.kdeplot, color="r", zorder=1, levels=6)
    # g1.plot_marginals(sns.rugplot, color="r", height=-.15, clip_on=False)
    ax1 = g1.ax_joint.get_xaxis()



    g1.ax_joint.set_xlabel(r'$\alpha_u$ (Sensitivity of Users)', fontsize=22)
    g1.ax_joint.set_ylabel(r'Activity of Users', fontsize=22)
    g1.savefig(os.path.join(save_fig_dir, savename1 + '.pdf'), format='pdf')
    plt.close()

    savename2 = "beta_popularity"
    g2 = sns.jointplot(data=df_b, x="beta", y="popularity",
                       marker="x", s=100, space=0, height=5)
    g2.plot_joint(sns.kdeplot, color="r", zorder=1, levels=6)
    g2.ax_joint.set_xlabel(r'$\beta_i$ (Unendurableness of Items)', fontsize=22)
    g2.ax_joint.set_ylabel(r'Popularity of Items', fontsize=22)
    g2.savefig(os.path.join(save_fig_dir, savename2 + '.pdf'), format='pdf')
    plt.close()

    a = 1
    # fig = plt.figure(figsize=(6, 3.3))
    #
    # gs = gridspec.GridSpec(1, 2)
    #
    # sns.set_style("ticks")
    # ax1 = fig.add_subplot(gs[0, 0])
    # g1 = sns.jointplot(data=df_a, x="alpha", y="popularity",ax=ax1)
    # ax2 = fig.add_subplot(gs[0, 1])
    # g2 = sns.jointplot(data=df_b, x="beta", y="popularity")
    #
    # mg0 = SeabornFig2Grid(g1, fig, gs[0])
    # mg1 = SeabornFig2Grid(g2, fig, gs[1])
    #
    # gs.tight_layout(fig)
    #
    #
    # # ax1.set_xlabel(r'$\alpha_u$', fontsize=12)
    # # ax1.set_ylabel(r'Popularity', fontsize=12)
    # # ax2.set_xlabel(r'$\beta_i$', fontsize=12)
    # # ax2.set_ylabel(r'Popularity', fontsize=12)
    #
    # # fig.savefig(os.path.join(save_fig_dir, savename + '.pdf'), format='pdf')
    #
    # fig.savefig(os.path.join(save_fig_dir, savename + '.pdf'), format='pdf', bbox_inches='tight')
    # plt.show()

def main(args):


    realpath = os.path.dirname(__file__)
    save_fig_dir = os.path.join(realpath, "figures")

    # create_dirs = [save_fig_dir]
    # create_dir(create_dirs)

    # filenames = walk_paths(result_dir)
    alpha_u, beta_i, df_big = loaddata(args)

    visual(alpha_u, beta_i, df_big, save_fig_dir)


if __name__ == '__main__':
    args = get_args()
    main(args)
