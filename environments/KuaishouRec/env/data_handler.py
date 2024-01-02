# -*- coding: utf-8 -*-
# @Time    : 2022/12/20 23:34
# @Author  : Chongming GAO
# @FileName: data_handler.py
import collections
import json
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

CODEPATH = os.path.dirname(__file__)
ROOTPATH = os.path.dirname(CODEPATH)
DATAPATH = os.path.join(ROOTPATH, "data")


def load_category():
    # load categories:
    print("load item feature")
    featurepath = os.path.join(DATAPATH, 'item_categories.json')
    with open(featurepath, 'r') as file:
        data_feat = json.load(file)
    print("number of items:", len(data_feat))
    list_feat = [0] * len(data_feat)
    for i in range(len(data_feat)):
        # list_feat[i] = set(data_feat[str(i)]['feature_index'])
        list_feat[i] = data_feat[str(i)]['feature_index']

    df_feat = pd.DataFrame(list_feat, columns=['feat0', 'feat1', 'feat2', 'feat3'])
    df_feat.index.name = "photo_id"
    df_feat[df_feat.isna()] = -1
    df_feat = df_feat + 1
    df_feat = df_feat.astype(int)

    return list_feat, df_feat

def get_lbe():
    if not os.path.isfile(os.path.join(DATAPATH, "user_id_small.csv")) or not os.path.isfile(
            os.path.join(DATAPATH, "item_id_small.csv")):
        small_path = os.path.join(DATAPATH, "small_matrix.csv")
        df_small = pd.read_csv(small_path, header=0, usecols=['user_id', 'photo_id'])

        user_id_small = pd.DataFrame(df_small["user_id"].unique(), columns=["user_id_small"])
        item_id_small = pd.DataFrame(df_small["photo_id"].unique(), columns=["item_id_small"])

        user_id_small.to_csv(os.path.join(DATAPATH, "user_id_small.csv"), index=False)
        item_id_small.to_csv(os.path.join(DATAPATH, "item_id_small.csv"), index=False)
    else:
        user_id_small = pd.read_csv(os.path.join(DATAPATH, "user_id_small.csv"))
        item_id_small = pd.read_csv(os.path.join(DATAPATH, "item_id_small.csv"))

    lbe_user = LabelEncoder()
    lbe_user.fit(user_id_small["user_id_small"])

    lbe_item = LabelEncoder()
    lbe_item.fit(item_id_small["item_id_small"])

    return lbe_user, lbe_item

def load_item_feat(only_small=False):
    list_feat, df_item = load_category()
    # video_mean_duration = load_video_duration()
    # df_item = df_feat.join(video_mean_duration, on=['item_id'], how="left")

    if only_small:
        lbe_user, lbe_item = get_lbe()
        item_list = lbe_item.classes_
        df_item_env = df_item.loc[item_list]
        return df_item_env

    return df_item

def get_df_kuairec(name="big_matrix.csv"):
    filename = os.path.join(DATAPATH, name)
    df_data = pd.read_csv(filename, usecols=['user_id', 'photo_id', 'watch_ratio'])


    list_feat, df_feat = load_category()

    if name == "big_matrix_processed.csv":
        only_small = False
    else:
        only_small = True
    df_item = load_item_feat(only_small)

    df_data = df_data.join(df_feat, on=['photo_id'], how="left")

    # if is_require_feature_domination:
    #     item_feat_domination = KuaiEnv.get_domination(df_data, df_item)
    #     return df_data, df_user, df_item, list_feat, item_feat_domination

    return df_data, df_item, list_feat



def get_sorted_domination_features(df_data, df_item, is_multi_hot, yname=None, threshold=None):
    item_feat_domination = dict()
    if not is_multi_hot: # for coat
        item_feat = df_item.columns.to_list()
        for x in item_feat:
            sorted_count = collections.Counter(df_data[x])
            sorted_percentile = dict(map(lambda x: (x[0], x[1] / len(df_data)), dict(sorted_count).items()))
            sorted_items = sorted(sorted_percentile.items(), key=lambda x: x[1], reverse=True)
            item_feat_domination[x] = sorted_items
    else: # for kuairec and kuairand
        df_item_filtered = df_item.filter(regex="^feat", axis=1)

        # df_item_flat = df_item_filtered.to_numpy().reshape(-1)
        # df_item_nonzero = df_item_flat[df_item_flat>0]

        feat_train = df_data.loc[df_data[yname] >= threshold, df_item_filtered.columns.to_list()]
        cats_train = feat_train.to_numpy().reshape(-1)
        pos_cat_train = cats_train[cats_train > 0]

        sorted_count = collections.Counter(pos_cat_train)
        sorted_percentile = dict(map(lambda x: (x[0], x[1] / sum(sorted_count.values())), dict(sorted_count).items()))
        sorted_items = sorted(sorted_percentile.items(), key=lambda x: x[1], reverse=True)

        item_feat_domination["feat"] = sorted_items

    return item_feat_domination

def get_training_item_domination():
    df_data, df_item, _ = get_df_kuairec("big_matrix.csv")
    CODEDIRPATH = os.path.dirname(__file__)
    feature_domination_path = os.path.join(CODEDIRPATH, "feature_domination.pickle")

    if os.path.isfile(feature_domination_path):
        item_feat_domination = pickle.load(open(feature_domination_path, 'rb'))
    else:
        item_feat_domination = get_sorted_domination_features(
            df_data, df_item, is_multi_hot=True, yname="watch_ratio",
            threshold=np.percentile(df_data["watch_ratio"], 80))
        pickle.dump(item_feat_domination, open(feature_domination_path, 'wb'))

    return item_feat_domination

