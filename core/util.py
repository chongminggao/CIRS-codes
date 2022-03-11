# -*- coding: utf-8 -*-
# @Time    : 2021/9/11 4:24 下午
# @Author  : Chongming GAO
# @FileName: util.py
import itertools
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
from numba import njit, jit
from scipy.sparse import csr_matrix, lil_matrix
from tqdm import tqdm

from core.inputs import SparseFeatP
from core.user_model import StaticDataset
from deepctr_torch.inputs import DenseFeat


def compute_action_distance(action: np.ndarray, actions_hist: np.ndarray,
                            env_name="VirtualTB-v0", realenv=None):  # for kuaishou data
    if env_name == "VirtualTB-v0":
        a = action - actions_hist
        if len(a.shape) > 1:
            dist = np.linalg.norm(a, axis=1)
        else:
            dist = np.linalg.norm(a)
    elif env_name == "KuaishouEnv-v0":
        # df_photo_env = realenv.df_photo_env
        # list_feat = realenv.list_feat
        # item_index = realenv.lbe_photo.inverse_transform([action])
        # item_index_hist = realenv.lbe_photo.inverse_transform(actions_hist)
        df_dist_small = realenv.df_dist_small

        dist = df_dist_small.iloc[action, actions_hist].to_numpy()

    return dist


def compute_exposure(t_diff: np.ndarray, dist: np.ndarray, tau):
    if tau <= 0:
        res = 0
        return res
    res = np.sum(np.exp(- t_diff * dist / tau))
    return res


def softplus_np(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def clip0(x):
    return np.amax(x, 0)


@njit
def compute_exposure_each_user(start_index: int,
                               distance_mat: np.ndarray,
                               timestamp: np.ndarray,
                               exposure_all: np.ndarray,
                               index_u: np.ndarray,
                               photo_u: np.ndarray,
                               tau: float
                               ):
    for i in range(1, len(index_u)):
        photo = photo_u[i]
        t_diff = timestamp[index_u[i]] - timestamp[start_index:index_u[i]]
        t_diff[t_diff == 0] = 1  # important!
        # dist_hist = np.fromiter(map(lambda x: distance_mat[x, photo], photo_u[:i]), np.float)

        dist_hist = np.zeros(i)
        for j in range(i):
            photo_j = photo_u[j]
            dist_hist[j] = distance_mat[photo_j, photo]

        exposure = np.sum(np.exp(- t_diff * dist_hist / tau))
        exposure_all[start_index + i] = exposure


def load_static_validate_data_kuaishou(entity_dim, feature_dim, DATAPATH):
    filename = os.path.join(DATAPATH, "small_matrix.csv")
    df_small = pd.read_csv(filename, usecols=['user_id', 'photo_id', 'watch_ratio', 'photo_duration'])
    df_small['photo_duration'] /= 1000

    featurepath = os.path.join(DATAPATH, 'item_categories.json')
    with open(featurepath, 'r') as file:
        data_feat = json.load(file)
    print("number of items:", len(data_feat))
    list_feat = [0] * len(data_feat)
    for i in range(len(data_feat)):
        # list_feat[i] = set(data_feat[str(i)]['feature_index'])
        list_feat[i] = data_feat[str(i)]['feature_index']

    df_feat = pd.DataFrame(list_feat, columns=['feat0', 'feat1', 'feat2', 'feat3'], dtype=int)
    df_feat.index.name = "photo_id"
    df_feat[df_feat.isna()] = -1
    df_feat = df_feat + 1
    df_feat = df_feat.astype(int)

    df_small = df_small.join(df_feat, on=['photo_id'], how="left")
    df_small.loc[df_small['watch_ratio'] > 5, 'watch_ratio'] = 5

    user_features = ["user_id"]
    item_features = ["photo_id"] + ["feat" + str(i) for i in range(4)] + ["photo_duration"]
    reward_features = ["watch_ratio"]

    col_names = user_features + item_features + reward_features

    df_x, df_y = df_small[user_features + item_features], df_small[reward_features]

    x_columns = [SparseFeatP("user_id", df_small['user_id'].max() + 1, embedding_dim=entity_dim)] + \
                [SparseFeatP("photo_id", df_small['photo_id'].max() + 1, embedding_dim=entity_dim)] + \
                [SparseFeatP("feat{}".format(i),
                             df_feat.max().max() + 1,
                             embedding_dim=feature_dim,
                             embedding_name="feat",  # Share the same feature!
                             padding_idx=0  # using padding_idx in embedding!
                             ) for i in range(4)] + \
                [DenseFeat("photo_duration", 1)]

    y_columns = [DenseFeat("y", 1)]

    photo_mean_duration_path = os.path.join(DATAPATH, "photo_mean_duration.json")
    with open(photo_mean_duration_path, 'r') as file:
        photo_mean_duration = json.load(file)
    photo_mean_duration = {int(k): v for k, v in photo_mean_duration.items()}

    dataset_val = StaticDataset(x_columns, y_columns, num_workers=4)
    dataset_val.compile_dataset(df_x, df_y)
    dataset_val.set_env_items(df_small, df_feat, photo_mean_duration)

    return dataset_val

def compute_exposure_effect_kuaishouRec(df_x, timestamp, list_feat, tau, MODEL_SAVE_PATH, DATAPATH):
    exposure_path = os.path.join(MODEL_SAVE_PATH, "..", "saved_exposure", "exposure_pos_{:.1f}.csv".format(tau))

    if os.path.isfile(exposure_path):
        print("loading saved exposure scores: ", exposure_path)
        exposure_pos_df = pd.read_csv(exposure_path)
        exposure_pos = exposure_pos_df.to_numpy()
        return exposure_pos

    similarity_mat = get_similarity_mat(list_feat, DATAPATH=DATAPATH)

    distance_mat = 1 / similarity_mat

    exposure_pos = np.zeros([len(df_x), 1])

    user_list = df_x["user_id"].unique()

    timestamp = timestamp.to_numpy()

    print("Compute the exposure effect (for the first time and will be saved for later usage)")
    for user in tqdm(user_list, desc="Computing exposure effect of historical data"):
        df_user = df_x[df_x['user_id'] == user]
        start_index = df_user.index[0]
        index_u = df_user.index.to_numpy()
        photo_u = df_user['photo_id'].to_numpy()
        compute_exposure_each_user(start_index, distance_mat, timestamp, exposure_pos,
                                   index_u, photo_u, tau)

    exposure_pos_df = pd.DataFrame(exposure_pos)

    if not os.path.exists(os.path.dirname(exposure_path)):
        os.mkdir(os.path.dirname(exposure_path))
    exposure_pos_df.to_csv(exposure_path, index=False)

    return exposure_pos

# For loading KuaishouRec Data
@njit
def find_negative(user_ids, photo_ids, mat_small, mat_big, df_negative, max_item):
    for i in range(len(user_ids)):
        user, item = user_ids[i], photo_ids[i]

        neg = item + 1
        while neg <= max_item:
            if neg == 1225:  # 1225 is an absent photo_id
                neg = 1226
            if mat_small[user, neg] or mat_big[user, neg]:
                neg += 1
            else:
                df_negative[i, 0] = user
                df_negative[i, 1] = neg
                break
        else:
            neg = item - 1
            while neg >= 0:
                if neg == 1225:  # 1225 is an absent photo_id
                    neg = 1224
                if mat_small[user, neg] or mat_big[user, neg]:
                    neg -= 1
                else:
                    df_negative[i, 0] = user
                    df_negative[i, 1] = neg
                    break


# @njit
# def find_negative(user_ids, photo_ids, mat_small, mat_big, mat_negative, max_item):
#     for i in range(len(user_ids)):
#         user, item = user_ids[i], photo_ids[i]
#
#         neg = item + 1
#         while neg <= max_item:
#             if neg == 1225:  # 1225 is an absent photo_id
#                 neg = 1226
#             if mat_small[user, neg] or mat_big[user, neg] or mat_negative[user, neg]:
#                 neg += 1
#             else:
#                 mat_negative[user, neg] = True
#                 break
#         else:
#             neg = item - 1
#             while neg >= 0:
#                 if neg == 1225:  # 1225 is an absent photo_id
#                     neg = 1224
#                 if mat_small[user, neg] or mat_big[user, neg] or mat_negative[user, neg]:
#                     neg -= 1
#                 else:
#                     mat_negative[user, neg] = True
#                     break

def get_distance_mat(list_feat, sub_index_list, DATAPATH="environments/KuaishouRec/data"):
    if sub_index_list is not None:
        distance_mat_small_path = os.path.join(DATAPATH, "distance_mat_photo_small.csv")
        if os.path.isfile(distance_mat_small_path):
            print("loading small distance matrix...")
            df_dist_small = pd.read_csv(distance_mat_small_path, index_col=0)
            df_dist_small.columns = df_dist_small.columns.astype(int)
            print("loading completed.")
        else:
            similarity_mat = get_similarity_mat(list_feat, DATAPATH)
            df_sim = pd.DataFrame(similarity_mat)
            df_sim_small = df_sim.loc[sub_index_list, sub_index_list]

            df_dist_small = 1.0 / df_sim_small

            df_dist_small.to_csv(distance_mat_small_path)

        return df_dist_small

    return None

def get_similarity_mat(list_feat, DATAPATH="environments/KuaishouRec/data"):
    similarity_mat_path = os.path.join(DATAPATH, "similarity_mat_photo.csv")
    if os.path.isfile(similarity_mat_path):
        # with open(similarity_mat_path, 'rb') as f:
        #     similarity_mat = np.load(f, allow_pickle=True, fix_imports=True)
        print("loading similarity matrix...")
        df_sim = pd.read_csv(similarity_mat_path, index_col=0)
        df_sim.columns = df_sim.columns.astype(int)
        print("loading completed.")
        similarity_mat = df_sim.to_numpy()
    else:
        series_feat_list = pd.Series(list_feat)
        df_feat_list = series_feat_list.to_frame("categories")
        df_feat_list.index.name = "photo_id"

        similarity_mat = np.zeros([len(df_feat_list), len(df_feat_list)])
        print("Compute the similarity matrix (for the first time and will be saved for later usage)")
        for i in tqdm(range(len(df_feat_list)), desc="Computing..."):
            for j in range(i):
                similarity_mat[i, j] = similarity_mat[j, i]
            for j in range(i, len(df_feat_list)):
                similarity_mat[i, j] = len(set(series_feat_list[i]).intersection(set(series_feat_list[j]))) / len(
                    set(series_feat_list[i]).union(set(series_feat_list[j])))

        df_sim = pd.DataFrame(similarity_mat)
        df_sim.to_csv(similarity_mat_path)

    return similarity_mat


# For loading KuaishouRec Data
def negative_sampling(df_big, df_feat, DATAPATH):
    small_path = os.path.join(DATAPATH, "small_matrix.csv")
    df_small = pd.read_csv(small_path, header=0, usecols=['user_id', 'photo_id'])

    mat_small = csr_matrix((np.ones(len(df_small)), (df_small['user_id'], df_small['photo_id'])),
                           shape=(df_big['user_id'].max() + 1, df_big['photo_id'].max() + 1), dtype=np.bool).toarray()
    # df_negative = df_big.copy()
    mat_big = csr_matrix((np.ones(len(df_big)), (df_big['user_id'], df_big['photo_id'])),
                         shape=(df_big['user_id'].max() + 1, df_big['photo_id'].max() + 1), dtype=np.bool).toarray()

    # mat_negative = lil_matrix((df_big['user_id'].max() + 1, df_big['photo_id'].max() + 1), dtype=np.bool).toarray()
    # find_negative(df_big['user_id'].to_numpy(), df_big['photo_id'].to_numpy(), mat_small, mat_big, mat_negative,
    #               df_big['photo_id'].max())
    # negative_pairs = np.array(list(zip(*mat_negative.nonzero())))
    # df_negative = pd.DataFrame(negative_pairs, columns=["user_id", "photo_id"])
    # df_negative = df_negative[df_negative['photo_id'] != 1225]  # 1225 is an absent photo_id

    df_negative = np.zeros([len(df_big), 2])
    find_negative(df_big['user_id'].to_numpy(), df_big['photo_id'].to_numpy(), mat_small, mat_big, df_negative,
                  df_big['photo_id'].max())

    df_negative = pd.DataFrame(df_negative, columns=["user_id", "photo_id"], dtype=int)
    df_negative = df_negative.merge(df_feat, on=['photo_id'], how='left')

    photo_mean_duration_path = os.path.join(DATAPATH, "photo_mean_duration.json")
    with open(photo_mean_duration_path, 'r') as file:
        photo_mean_duration = json.load(file)
    photo_mean_duration = {int(k): v for k, v in photo_mean_duration.items()}

    df_negative['photo_duration'] = df_negative['photo_id'].map(lambda x: photo_mean_duration[x])
    df_negative['watch_ratio'] = 0.0

    return df_negative
