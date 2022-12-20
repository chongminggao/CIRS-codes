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
import traceback

import gym
import torch
from gym import register
from torch import nn

from core.inputs import SparseFeatP
from core.user_model_pairwise import UserModel_Pairwise
from core.util import negative_sampling, load_static_validate_data_kuaishou
from deepctr_torch.inputs import DenseFeat
import pandas as pd
import numpy as np

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

    parser.add_argument("--num_trajectory", type=int, default=200)
    parser.add_argument("--force_length", type=int, default=10)
    parser.add_argument('--epsilon', default=0, type=float)

    parser.add_argument('--l2_reg_dnn', default=0.1, type=float)

    parser.add_argument("--feature_dim", type=int, default=16)
    parser.add_argument("--entity_dim", type=int, default=16)
    parser.add_argument("--user_model_name", type=str, default="DeepFM-IPS-pairwise")
    parser.add_argument('--dnn', default=(64, 64), type=int, nargs="+")
    parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--cuda', default=1, type=int)
    # # env special:
    parser.add_argument('--leave_threshold', default=1, type=float)
    parser.add_argument('--num_leave_compute', default=5, type=int)

    parser.add_argument("--message", type=str, default="DeepFM-IPS-pairwise")

    args = parser.parse_known_args()[0]
    return args


def compute_IPS_kuaishouRec(df_x_all, df_big) -> np.ndarray:
    IPS_photo = collections.Counter(df_big['photo_id'])
    IPS_data = df_x_all['photo_id'].map(lambda x: IPS_photo[x])
    IPS_data[IPS_data == 0] = 1
    IPS_data = 1.0 / IPS_data
    IPS_data_np = IPS_data.to_frame().to_numpy()

    return IPS_data_np


def load_dataset_kuaishou_IPS_pairwise(entity_dim, feature_dim):
    # CODEPATH = os.path.dirname(__file__)
    # ROOTPATH = os.path.dirname(CODEPATH)
    # DATAPATH = os.path.join(ROOTPATH, "data")

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

    x_columns = [SparseFeatP("user_id", df_big['user_id'].max() + 1, embedding_dim=entity_dim)] + \
                [SparseFeatP("photo_id", df_big['photo_id'].max() + 1, embedding_dim=entity_dim)] + \
                [SparseFeatP("feat{}".format(i),
                             df_feat.max().max() + 1,
                             embedding_dim=feature_dim,
                             embedding_name="feat",  # Share the same feature!
                             padding_idx=0  # using padding_idx in embedding!
                             ) for i in range(4)] + \
                [DenseFeat("photo_duration", 1)]

    y_columns = [DenseFeat("y", 1)]

    df_negative = negative_sampling(df_big, df_feat, DATAPATH)
    df_x_neg, df_y_neg = df_negative[user_features + item_features], df_negative[reward_features]

    df_x_neg = df_x_neg.rename(columns={k: k + "_neg" for k in df_x_neg.columns.to_numpy()})

    df_x_all = pd.concat([df_x, df_x_neg], axis=1)

    IPS_data = compute_IPS_kuaishouRec(df_x_all, df_big)

    dataset = StaticDataset(x_columns, y_columns, num_workers=4)
    dataset.compile_dataset(df_x_all, df_y, IPS_data)

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
    static_dataset, x_columns, y_columns = load_dataset_kuaishou_IPS_pairwise(args.entity_dim, args.feature_dim)

    dataset_val = load_static_validate_data_kuaishou(args.entity_dim, args.feature_dim, DATAPATH)

    # %% 4. Setup model
    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")

    SEED = 2021
    task = "regression"
    task_logit_dim = 1
    model = UserModel_Pairwise(x_columns, y_columns, task, task_logit_dim,
                           dnn_hidden_units=args.dnn, seed=SEED, l2_reg_dnn=args.l2_reg_dnn,
                           device=device)

    model.compile(optimizer="adam",
                  # loss_dict=task_loss_dict,
                  loss_func=loss_kuaishou_IPS_pairwise,
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
                          force_length=args.force_length))

    # %% 5. Learn model
    history = model.fit_data(static_dataset, dataset_val,
                             batch_size=args.batch_size, epochs=args.epoch,
                             callbacks=[LoggerCallback_Update(logger_path)])
    logger.info(history.history)

    model_parameters = {"feature_columns": x_columns, "y_columns": y_columns, "task": task,
                        "task_logit_dim": task_logit_dim, "dnn_hidden_units": args.dnn, "seed": SEED, "device": device}

    model_parameter_path = os.path.join(MODEL_SAVE_PATH,
                                        "{}_params_{}.pickle".format(args.user_model_name, args.message))
    with open(model_parameter_path, "wb") as output_file:
        pickle.dump(model_parameters, output_file)

    normed_mat = KuaishouEnv.compute_normed_reward(model, lbe_user, lbe_photo, df_photo_env)
    mat_save_path = os.path.join(MODEL_SAVE_PATH, "normed_mat-{}.pickle".format(args.message))
    with open(mat_save_path, "wb") as f:
        pickle.dump(normed_mat, f)

    #  To cpu
    user_model = model.cpu()
    user_model.linear_model.device = "cpu"
    user_model.linear.device = "cpu"
    # for linear_model in user_model.linear_model_task:
    #     linear_model.device = "cpu"

    model_save_path = os.path.join(MODEL_SAVE_PATH, "{}_{}.pt".format(args.user_model_name, args.message))
    torch.save(model.state_dict(), model_save_path)

    REMOTE_ROOT = "/root/Counterfactual_IRS"
    LOCAL_PATH = logger_path
    REMOTE_PATH = os.path.join(REMOTE_ROOT, os.path.dirname(LOCAL_PATH))

    # my_upload(LOCAL_PATH, REMOTE_PATH, REMOTE_ROOT)



sigmoid = nn.Sigmoid()
def loss_kuaishou_IPS_pairwise(y, y_deepfm_pos, y_deepfm_neg, IPS_score):
    loss_y = (((y_deepfm_pos - y) ** 2) * IPS_score).mean()

    # bpr_click = - (sigmoid(y_deepfm_pos - y_deepfm_neg).log() * IPS_score).mean()
    # loss = loss_y + bpr_click
    
    loss = loss_y

    return loss


if __name__ == '__main__':
    args = get_args()
    try:
        main(args)
    except Exception as e:
        var = traceback.format_exc()
        print(var)
        logzero.logger.error(var)
