# -*- coding: utf-8 -*-
# @Time    : 2021/7/26 10:15 上午
# @Author  : Chongming GAO
# @FileName: train_staticRS_on_logData_evaluate_in_realEnv.py


import argparse
import collections
import datetime
import functools

import json
import os
import pickle
import time

import gym
import torch
import tqdm
from gym import register
from torch import nn

from core.inputs import SparseFeatP
from core.user_model_DICE import UserModel_DICE
from core.util import compute_exposure_each_user, negative_sampling, get_similarity_mat, \
    load_static_validate_data_kuaishou
from deepctr_torch.inputs import DenseFeat
import pandas as pd
import numpy as np
from tensorflow.python.keras.callbacks import Callback

from core.static_dataset import StaticDataset

import logzero
from logzero import logger

from environments.KuaishouRec.env.data_handler import get_training_item_domination
from environments.KuaishouRec.env.kuaishouEnv import KuaishouEnv
from evaluation import test_kuaishou, test_static_model_in_RL_env
# from util.upload import my_upload
from util.utils import create_dir, LoggerCallback_Update

DATAPATH = "environments/KuaishouRec/data"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action="store_true")
    parser.add_argument("--env", type=str, default='KuaishouEnv-v0')


    # recommendation related:
    # parser.add_argument('--not_softmax', action="store_false")
    parser.add_argument('--is_softmax', dest='is_softmax', action='store_true')
    parser.add_argument('--not_softmax', dest='is_softmax', action='store_false')
    parser.set_defaults(is_softmax=True)

    parser.add_argument('--l2_reg_dnn', default=0.1, type=float)

    parser.add_argument("--num_trajectory", type=int, default=200)
    parser.add_argument("--force_length", type=int, default=10)
    parser.add_argument('--epsilon', default=0, type=float)
    parser.add_argument("--top_rate", type=float, default=0.6)

    parser.add_argument("--feature_dim", type=int, default=16)
    parser.add_argument("--entity_dim", type=int, default=16)
    parser.add_argument("--user_model_name", type=str, default="DICE")
    parser.add_argument('--dnn', default=(64, 64), type=int, nargs="+")
    parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--cuda', default=1, type=int)
    # # env special:
    parser.add_argument('--leave_threshold', default=1, type=float)
    parser.add_argument('--num_leave_compute', default=5, type=int)
    # exposure parameters:
    # parser.add_argument('--tau', default=3600, type=float)

    parser.add_argument("--message", type=str, default="No special")

    args = parser.parse_known_args()[0]
    return args


def compute_popularity_kuaishouRec(df_x, df_x_neg, df_big) -> (np.ndarray, np.ndarray):
    count = collections.Counter(df_big['photo_id'])
    popularity = df_x['photo_id'].map(lambda x: count[x])
    popularity.loc[popularity == 0] = 1

    popularity_neg = df_x_neg['photo_id'].map(lambda x: count[x])
    popularity_neg.loc[popularity_neg == 0] = 1

    pop_pos, pop_neg = popularity.to_frame().to_numpy(), popularity_neg.to_frame().to_numpy()

    return pop_pos, pop_neg


def load_dataset_kuaishou_DICE(entity_dim, feature_dim, MODEL_SAVE_PATH):
    filename = os.path.join(DATAPATH, "big_matrix.csv")
    df_big = pd.read_csv(filename, usecols=['user_id', 'photo_id', 'timestamp', 'watch_ratio', 'photo_duration'])
    df_big['photo_duration'] /= 1000

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

    df_big = df_big.join(df_feat, on=['photo_id'], how="left")
    df_big.loc[df_big['watch_ratio'] > 5, 'watch_ratio'] = 5

    user_features = ["user_id"]
    item_features = ["photo_id"] + ["feat" + str(i) for i in range(4)] + ["photo_duration"]
    reward_features = ["watch_ratio"]

    df_x, df_y = df_big[user_features + item_features], df_big[reward_features]

    x_columns = [SparseFeatP("user_id_int", df_big['user_id'].max() + 1, embedding_dim=entity_dim,
                             embedding_name="user_int")] + \
                [SparseFeatP("user_id_con", df_big['user_id'].max() + 1, embedding_dim=entity_dim,
                             embedding_name="user_con")] + \
                [SparseFeatP("photo_id_int", df_big['photo_id'].max() + 1, embedding_dim=entity_dim,
                             embedding_name="photo_int")] + \
                [SparseFeatP("photo_id_con", df_big['photo_id'].max() + 1, embedding_dim=entity_dim,
                             embedding_name="photo_con")] + \
                [SparseFeatP("feat{}".format(i),
                             df_feat.max().max() + 1,
                             embedding_dim=feature_dim,
                             embedding_name="feat",  # Share the same feature!
                             padding_idx=0  # using padding_idx in embedding!
                             ) for i in range(4)] + \
                [DenseFeat("photo_duration", 1)] + \
                [SparseFeatP("photo_id_int_neg", df_big['photo_id'].max() + 1, embedding_dim=entity_dim,
                             embedding_name="photo_int")] + \
                [SparseFeatP("photo_id_con_neg", df_big['photo_id'].max() + 1, embedding_dim=entity_dim,
                             embedding_name="photo_con")] + \
                [SparseFeatP("feat{}_neg".format(str(i)),
                             df_feat.max().max() + 1,
                             embedding_dim=feature_dim,
                             embedding_name="feat",  # Share the same feature!
                             padding_idx=0  # using padding_idx in embedding!
                             ) for i in range(4)] + \
                [DenseFeat("photo_duration_neg", 1)]

    y_columns = [DenseFeat("y", 1)]

    # timestamp = df_big['timestamp']

    df_negative = negative_sampling(df_big, df_feat, DATAPATH)
    df_x_neg, df_y_neg = df_negative[user_features + item_features], df_negative[reward_features]

    feature_user = ['user_id_int', 'user_id_con']
    features_pos = ["photo_id_int", "photo_id_con"] + \
                   ["feat" + str(i) for i in range(4)] + ["photo_duration"]
    features_neg = ["photo_id_int_neg", "photo_id_con_neg"] + \
                   ["feat{}_neg".format(str(i)) for i in range(4)] + ["photo_duration_neg"]

    df_x_new = df_x[item_features]
    df_x_new['user_id_int'] = df_x['user_id']
    df_x_new['user_id_con'] = df_x['user_id']
    df_x_new['photo_id_int'] = df_x['photo_id']
    df_x_new = df_x_new.rename(columns={"photo_id": "photo_id_con"})

    df_x_neg_new = df_x_neg[item_features]
    df_x_neg_new['photo_id_int'] = df_x_neg_new['photo_id']
    df_x_neg_new = df_x_neg_new.rename(columns={"photo_id": "photo_id_con"})
    df_x_neg_new = df_x_neg_new.rename(columns={k: k + "_neg" for k in df_x_neg_new.columns.to_numpy()})

    df_x_new = df_x_new[feature_user + features_pos]
    df_x_neg_new = df_x_neg_new[features_neg]

    df_x_all = pd.concat([df_x_new, df_x_neg_new], axis=1)

    pop_pos, pop_neg = compute_popularity_kuaishouRec(df_x, df_x_neg, df_big)
    is_conformity_larger = (pop_pos > pop_neg).astype(int)
    is_conformity_larger[is_conformity_larger == 0] = -1

    dataset = StaticDataset(x_columns, y_columns, num_workers=4)
    dataset.compile_dataset(df_x_all, df_y, is_conformity_larger)

    return dataset, x_columns, y_columns

def main(args):
    args.entity_dim = args.feature_dim
    # %% 1. Create dirs
    MODEL_SAVE_PATH = os.path.join(".", "saved_models", args.env, args.user_model_name)

    create_dirs = [os.path.join(".", "saved_models"),
                   os.path.join(".", "saved_models", args.env),
                   MODEL_SAVE_PATH,
                   os.path.join(MODEL_SAVE_PATH, "logs")]
    create_dir(create_dirs)

    nowtime = datetime.datetime.fromtimestamp(time.time()).strftime("%Y_%m_%d-%H_%M_%S")
    logger_path = os.path.join(MODEL_SAVE_PATH, "logs", "[{}]_{}.log".format(args.message, nowtime))
    logzero.logfile(logger_path)
    logger.info(json.dumps(vars(args), indent=2))

    # %% 2. Prepare Envs
    mat, lbe_user, lbe_photo, list_feat, df_photo_env, df_dist_small = KuaishouEnv.load_mat()
    register(
        id=args.env,  # 'KuaishouEnv-v0',
        entry_point='environments.KuaishouRec.env.kuaishouEnv:KuaishouEnv',
        kwargs={"mat": mat,
                "lbe_user": lbe_user,
                "lbe_photo": lbe_photo,
                "num_leave_compute": args.num_leave_compute,
                "leave_threshold": args.leave_threshold,
                "list_feat": list_feat,
                "df_photo_env": df_photo_env,
                "df_dist_small": df_dist_small}
    )
    env = gym.make(args.env)

    # %% 3. Prepare dataset
    static_dataset, x_columns, y_columns = load_dataset_kuaishou_DICE(args.entity_dim, args.feature_dim,
                                                                      MODEL_SAVE_PATH)

    dataset_val = load_static_validate_data_kuaishou(args.entity_dim, args.feature_dim, DATAPATH)

    # %% 4. Setup model
    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")

    SEED = 2021
    tasks = "regression"
    task_logit_dim = 1
    model = UserModel_DICE(x_columns, y_columns, tasks, task_logit_dim,
                           dnn_hidden_units=args.dnn, seed=SEED, l2_reg_dnn=args.l2_reg_dnn,
                           device=device)

    model.compile(optimizer="adam",
                  # loss_dict=task_loss_dict,
                  loss_func=loss_kuaishou_DICE,
                  metric_fun={"mae": lambda y, y_predict: nn.functional.l1_loss(torch.from_numpy(y),
                                                                                torch.from_numpy(y_predict)).numpy(),
                              "mse": lambda y, y_predict: nn.functional.mse_loss(torch.from_numpy(y),
                                                                                 torch.from_numpy(y_predict)).numpy()},
                  metrics=None)  # No evaluation step at offline stage

    item_feat_domination = get_training_item_domination()
    model.compile_RL_test(
        functools.partial(test_static_model_in_RL_env, env=env, dataset_val=dataset_val, is_softmax=args.is_softmax,
                          epsilon=args.epsilon, is_ucb=False, need_transform=True,
                          num_trajectory=args.num_trajectory, item_feat_domination=item_feat_domination,
                          force_length=args.force_length, top_rate=args.top_rate))

    # %% 5. Learn model
    history = model.fit_data(static_dataset, dataset_val,
                             batch_size=args.batch_size, epochs=args.epoch,
                             callbacks=[LoggerCallback_Update(logger_path)])
    logger.info(history.history)

    model_parameters = {"feature_columns": x_columns, "y_columns": y_columns, "num_tasks": len(tasks), "tasks": tasks,
                        "task_logit_dim": task_logit_dim, "dnn_hidden_units": args.dnn, "seed": SEED, "device": device}

    model_parameter_path = os.path.join(MODEL_SAVE_PATH,
                                        "{}_params_{}.pickle".format(args.user_model_name, args.message))
    with open(model_parameter_path, "wb") as output_file:
        pickle.dump(model_parameters, output_file)

    REMOTE_ROOT = "/root/Counterfactual_IRS"
    LOCAL_PATH = logger_path
    REMOTE_PATH = os.path.join(REMOTE_ROOT, os.path.dirname(LOCAL_PATH))

    # my_upload(LOCAL_PATH, REMOTE_PATH, REMOTE_ROOT)





sigmoid = nn.Sigmoid()


def loss_kuaishou_DICE(y, y_deepfm_pos, y_deepfm_neg,
                       y_deepfm_pos_int, y_deepfm_neg_int,
                       y_deepfm_pos_con, y_deepfm_neg_con, score):
    loss_y = ((y_deepfm_pos - y) ** 2).mean()
    bpr_click = - sigmoid(y_deepfm_pos - y_deepfm_neg).log().mean()
    bpr_con = - (sigmoid(y_deepfm_pos_con - y_deepfm_neg_con).log() * score).mean()
    bpr_int = - (sigmoid(y_deepfm_pos_int - y_deepfm_neg_int).log() * (score < 0)).mean()
    loss1 = loss_y + bpr_click + bpr_con + bpr_int
    # loss2 = (y_deepfm_neg**2 + y_deepfm_pos_int**2 + y_deepfm_neg_int**2 + y_deepfm_pos_con**2 + y_deepfm_neg_con**2).mean()
    # loss = loss1 + loss2
    loss = loss1
    # if bool(loss1.isinf()) or bool((-loss1).isinf()):
    #     a = 1
    return loss


if __name__ == '__main__':
    args = get_args()
    main(args)
