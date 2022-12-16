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
import time
import traceback

import gym
from gym.envs.registration import register
import torch

from deepctr_torch.inputs import DenseFeat
import pandas as pd
from keras.callbacks import Callback

from core.user_model import StaticDataset
from core.user_model_mmoe import UserModel_MMOE
import logzero
from logzero import logger

from evaluation import test_taobao
# from util.upload import my_upload
from util.utils import create_dir, LoggerCallback_Update


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action="store_true")
    parser.add_argument("--env", type=str, default="VirtualTB-v0")
    parser.add_argument("--feature_dim", type=int, default=4)
    parser.add_argument("--model_name", type=str, default="MLP")
    parser.add_argument('--dnn', default=(256, 256), type=int, nargs="+")
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--cuda', default=0, type=int)

    # env special:
    parser.add_argument('--leave_threshold', default=1.0, type=float)
    parser.add_argument('--num_leave_compute', default=5, type=int)
    parser.add_argument('--max_turn', default=50, type=int)

    parser.add_argument("--message", type=str, default="MLP")
    # parser.add_argument('--dim', default=20, type=int)

    args = parser.parse_known_args()[0]
    return args


def load_dataset_virtualTaobao(feature_dim=10):
    filename = "environments/VirtualTaobao/virtualTB/SupervisedLearning/dataset.txt"
    user_features = ["feat" + str(i) for i in range(91)]
    item_features = ["y" + str(i) for i in range(27)]
    reward_features = ["click"]

    col_names = user_features + item_features + reward_features
    df = pd.read_csv(filename, header=None, sep="\s|,", names=col_names, engine='python')
    df_x, df_y = df[user_features], df[item_features + reward_features]

    # x_columns = [SparseFeatP(feat, 2, embedding_dim=feature_dim)  # Note there is no mask for missing value
    #              for feat in user_features[:88]] + \
    #             [DenseFeat(feat, 1) for feat in user_features[88:]]
    x_columns = [DenseFeat("feat_user", 91)]

    y_columns = [DenseFeat("feat_item", 27)] + [DenseFeat("y", 1)]

    dataset = StaticDataset(x_columns, y_columns, num_workers=4)
    dataset.compile_dataset(df_x, df_y)

    return dataset, x_columns, y_columns


def main(args):
    # 1. Create dirs
    MODEL_SAVE_PATH = os.path.join(".", "saved_models", args.env, args.model_name)

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
    # import virtualTB
    register(
        id=args.env,  # 'VirtualTB-v0',
        entry_point='environments.VirtualTaobao.virtualTB.envs:VirtualTB',
        kwargs={"num_leave_compute": args.num_leave_compute,
                "leave_threshold": args.leave_threshold,
                "max_turn": args.max_turn + 1}
    )

    env = gym.make('VirtualTB-v0')
    env.set_state_mode(True)  # return the states as user initial profile vectors.

    # %% 3. Prepare dataset
    static_dataset, x_columns, y_columns = load_dataset_virtualTaobao(feature_dim=args.feature_dim)

    # %% 4. Setup model
    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")

    SEED = 2022

    # tasks = "regression"
    # task_loss = collections.OrderedDict({feat.name: "mse" for feat in y_columns})
    # model = UserModel_DeepFM(x_columns, y_columns, tasks,
    #                          dnn_hidden_units=args.dnn, seed=SEED,
    #                          device=device)

    tasks = collections.OrderedDict({feat.name: "regression" for feat in y_columns})
    # task_loss_dict = collections.OrderedDict({feat.name: "mse" for feat in y_columns})

    task_logit_dim = {feat.name: feat.dimension if isinstance(feat, DenseFeat) else feat.embedding_dim for feat in
                      y_columns}
    model = UserModel_MMOE(x_columns, y_columns, len(tasks), tasks, task_logit_dim,
                           dnn_hidden_units=args.dnn, seed=SEED,
                           device=device)

    model.compile(optimizer="adam",
                  # loss_dict=task_loss_dict,
                  loss_func=loss_taobao,
                  metrics=None)  # No evaluation step at offline stage

    model.compile_RL_test(functools.partial(test_taobao, env=env))

    # %% 5. Learn model
    history = model.fit_data(static_dataset,
                             batch_size=args.batch_size, epochs=args.epoch,
                             callbacks=[[LoggerCallback_Update(logger_path)]])
    logger.info(history.history)

    REMOTE_ROOT = "/root/Counterfactual_IRS"
    LOCAL_PATH = logger_path
    REMOTE_PATH = os.path.join(REMOTE_ROOT, os.path.dirname(LOCAL_PATH))

    # my_upload(LOCAL_PATH, REMOTE_PATH, REMOTE_ROOT)





def loss_taobao(y_predict, y_true, exposure, y_index):
    loss = 0
    click = y_true[:, -1].unsqueeze(-1)
    loss_func = torch.nn.functional.mse_loss

    for yname, yind in y_index.items():

        # Opition 1: both action and click is mask by y
        # loss_i = loss_func(click * y_predict[:, yind[0]:yind[1]], click * y[:, yind[0]:yind[1]], reduction="mean") # For taobao_dataset, only training on positive states.

        # Opition 2: only action is mask by y
        if yname == "y":
            loss_i = loss_func(y_predict[:, yind[0]:yind[1]], y_true[:, yind[0]:yind[1]], reduction="mean")
        else:
            loss_i = loss_func(click * y_predict[:, yind[0]:yind[1]], click * y_true[:, yind[0]:yind[1]],
                               reduction="mean")
        loss += loss_i

    return loss


if __name__ == '__main__':
    args = get_args()
    try:
        main(args)
    except Exception as e:
        var = traceback.format_exc()
        print(var)
        logzero.logger.error(var)
