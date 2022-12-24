# -*- coding: utf-8 -*-
# @Time    : 2021/9/13 10:25 上午
# @Author  : Chongming GAO
# @FileName: visual_RL.py

import argparse
import collections
import os
import pickle

from collections import OrderedDict


import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

from core.user_model_pairwise import UserModel_Pairwise


DATAPATH = "../environments/KuaishouRec/data"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual_path", type=str, default="results_alpha_beta")
    parser.add_argument("--user_model_name", type=str, default="DeepFM")
    parser.add_argument("--read_message", type=str, default="Pair11")
    parser.add_argument('--is_ab', dest='is_ab', action='store_true')
    parser.add_argument('--no_ab', dest='is_ab', action='store_false')
    parser.set_defaults(is_ab=True)

    args = parser.parse_known_args()[0]
    return args


def loaddata(model_parameter_path, model_save_path):
    with open(model_parameter_path, "rb") as file:
        model_params = pickle.load(file)

    model_params["device"] = "cpu"
    user_model = UserModel_Pairwise(**model_params)
    user_model.load_state_dict(torch.load(model_save_path))

    with open(model_parameter_path, "rb") as file:
        model_params = pickle.load(file)

    # if hasattr(user_model, 'ab_embedding_dict') and is_ab:
    alpha_u = user_model.ab_embedding_dict["alpha_u"].weight.detach().cpu().numpy()
    beta_i = user_model.ab_embedding_dict["beta_i"].weight.detach().cpu().numpy()
    # else:
    #     print("Note there are no available alpha and beta！！")
    #     alpha_u = np.ones([7176, 1])
    #     beta_i = np.ones([10729, 1])

    filename = os.path.join(DATAPATH, "big_matrix.csv")
    df_big = pd.read_csv(filename, usecols=['user_id', 'photo_id', 'timestamp', 'watch_ratio', 'photo_duration'])

    return alpha_u, beta_i, df_big


def visual(alpha_u, beta_i, df_big, save_fig_dir, savename="alpha_beta"):
    user_cnt = collections.Counter(df_big['user_id'])
    item_cnt = collections.Counter(df_big['photo_id'])

    lbe_user = LabelEncoder()
    lbe_user.fit(np.array(list(user_cnt.keys())))
    lbe_item = LabelEncoder()
    lbe_item.fit(np.array(list(item_cnt.keys())))

    item_cnt[1225] = 0  # 1225 is missing from the item sets
    user_cnt = OrderedDict(sorted(user_cnt.items(), key=lambda x: x[0]))
    item_cnt = OrderedDict(sorted(item_cnt.items(), key=lambda x: x[0]))

    df_a = pd.DataFrame({"alpha": alpha_u.squeeze(), "popularity": user_cnt.values()})
    df_b = pd.DataFrame({"beta": beta_i.squeeze(), "popularity": item_cnt.values()})

    df_a = df_a[df_a["alpha"] < 0.06]
    df_b = df_b[df_b["beta"] < 0.08]

    # sns.jointplot(x=alpha_u.squeeze(), y=np.array(list(user_cnt.values())), kind="kde")
    # g1.savefig(os.path.join(save_fig_dir, savename + '.pdf'), format='pdf')
    # plt.scatter(alpha_u, user_cnt.values())
    # plt.scatter(beta_i, item_cnt.values())

    # https://stackoverflow.com/questions/34706845/change-xticklabels-fontsize-of-seaborn-heatmap
    # https://seaborn.pydata.org/generated/seaborn.set_theme.html#seaborn.set_theme
    sns.set(style="ticks", font_scale=1.5)
    savename1 = "alpha_popularity"
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



def visual_alpha_beta(visual_path, user_model_name, read_message):
    realpath = os.path.dirname(__file__)
    save_fig_dir = os.path.join(realpath, "figures")

    model_parameter_path = os.path.join(visual_path,
                                        "{}_params_{}.pickle".format(user_model_name, read_message))
    model_save_path = os.path.join(visual_path, "{}_{}.pt".format(user_model_name, read_message))

    alpha_u, beta_i, df_big = loaddata(model_parameter_path, model_save_path)

    visual(alpha_u, beta_i, df_big, save_fig_dir)


if __name__ == '__main__':

    visual_path = "results_alpha_beta"
    user_model_name = "DeepFM"
    read_message = "Pair11"

    visual_alpha_beta(visual_path, user_model_name, read_message)
