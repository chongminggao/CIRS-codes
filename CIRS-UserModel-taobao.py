# -*- coding: utf-8 -*-
# @Time    : 2021/7/26 10:15 上午
# @Author  : Chongming GAO
# @FileName: train_staticRS_on_logData_evaluate_in_realEnv.py


import argparse
import collections
import datetime
import json
import os
import pickle
import time

import torch
import tqdm

from core.util import compute_action_distance, compute_exposure
from deepctr_torch.inputs import DenseFeat
import pandas as pd
import numpy as np
from tensorflow.python.keras.callbacks import Callback

from core.user_model import StaticDataset
from core.user_model_mmoe import UserModel_MMOE
import logzero
from logzero import logger

from util.utils import create_dir, LoggerCallback_Update


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action="store_true")
    parser.add_argument("--env", type=str, default='VirtualTB-v0')
    parser.add_argument("--feature_dim", type=int, default=4)
    parser.add_argument("--user_model_name", type=str, default="MLP")
    parser.add_argument('--dnn', default=(64, 64), type=int, nargs="+")
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--epoch', default=3, type=int)
    parser.add_argument('--cuda', default=0, type=int)
    # # env:
    # parser.add_argument('--leave_threshold', default=4.0, type=float)
    # parser.add_argument('--num_leave_compute', default=5, type=int)
    # exposure parameters:
    parser.add_argument('--tau', default=0.01, type=float)

    parser.add_argument("--message", type=str, default="UserModel1")


    # parser.add_argument('--dim', default=20, type=int)

    args = parser.parse_known_args()[0]
    return args


def compute_exposure_effect_virtualTaobao(df_x, tau):
    exposure_all = np.zeros([len(df_x), 1])

    timestamp = df_x["feat90"].to_numpy().astype(int)
    action = df_x[["y{}".format(i) for i in range(27)]].to_numpy()

    for ind in tqdm.tqdm(range(len(df_x)), desc="Computing exposure effect of historical data"):
        if timestamp[ind] == 1:
            start = ind
        else:

            action_history = action[start:ind, :]
            action_ind = action[ind, :]
            distance = compute_action_distance(action_ind, action_history)

            t_diff = (ind - start) - np.arange(ind - start)
            exposure = compute_exposure(t_diff, distance, tau)
            exposure_all[ind] = exposure
    return exposure_all


def load_dataset_virtualTaobao(tau, feature_dim=10):
    # import virtualTB
    # register(
    #     id=args.env,  # 'VirtualTB-v0',
    #     entry_point='environments.VirtualTaobao.virtualTB.envs:VirtualTB',
    #     kwargs={"num_leave_compute": args.num_leave_compute,
    #             "leave_threshold": args.leave_threshold}
    # )
    #
    # env = gym.make('VirtualTB-v0')
    # env.set_state_mode(True)  # return the states as user initial profile vectors.

    filename = "environments/VirtualTaobao/virtualTB/SupervisedLearning/dataset.txt"
    user_features = ["feat" + str(i) for i in range(91)]
    item_features = ["y" + str(i) for i in range(27)]
    reward_features = ["click"]

    col_names = user_features + item_features + reward_features
    df = pd.read_csv(filename, header=None, sep="\s|,", names=col_names, engine='python')
    df_x, df_y = df[user_features + item_features], df[reward_features]

    # x_columns = [SparseFeatP(feat, 2, embedding_dim=feature_dim)  # Note there is no mask for missing value
    #              for feat in user_features[:88]] + \
    #             [DenseFeat(feat, 1) for feat in user_features[88:]] + \
    #             [DenseFeat("feat_item", 27)]
    #             [DenseFeat("exposure", 1)] # Attention, No exposure as feature!!!

    x_columns = [DenseFeat("user_feat", 91)] + [DenseFeat("feat_item", 27)]
    # + [DenseFeat("exposure", 1)] # Attention, No exposure as feature!!!

    y_columns = [DenseFeat("y", 1)]

    exposure_all = compute_exposure_effect_virtualTaobao(df_x, tau)

    dataset = StaticDataset(x_columns, y_columns, user_features, item_features, num_workers=4)
    dataset.compile_dataset(df_x, df_y, exposure_all)

    return dataset, x_columns, y_columns


def main(args):
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

    # %% 2. Prepare dataset
    static_dataset, x_columns, y_columns = load_dataset_virtualTaobao(args.tau, feature_dim=args.feature_dim)

    # %% 3. Setup model
    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")

    SEED = 2021
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

    # model.compile_RL_test(functools.partial(test_taobao, env=env))

    # %% 4. Learn the model
    history = model.fit_data(static_dataset,
                             batch_size=args.batch_size, epochs=args.epoch,
                             callbacks=[[LoggerCallback_Update(logger_path)]])
    logger.info(history.history)

    model_parameters = {"feature_columns": x_columns, "y_columns": y_columns, "num_tasks": len(tasks), "tasks": tasks,
                        "task_logit_dim": task_logit_dim, "dnn_hidden_units": args.dnn, "seed": SEED, "device": device}

    model_parameter_path = os.path.join(MODEL_SAVE_PATH, "{}_params_{}.pickle".format(args.user_model_name, args.message))
    with open(model_parameter_path, "wb") as output_file:
        pickle.dump(model_parameters, output_file)

    #  To cpu
    model = model.cpu()
    model.linear_model.device = "cpu"
    for linear_model in model.linear_model_task:
        linear_model.device = "cpu"

    model_save_path = os.path.join(MODEL_SAVE_PATH, "{}_{}.pt".format(args.user_model_name, args.message))
    torch.save(model.state_dict(), model_save_path)

    REMOTE_ROOT = "/root/Counterfactual_IRS"
    LOCAL_PATH = logger_path
    REMOTE_PATH = os.path.join(REMOTE_ROOT, os.path.dirname(LOCAL_PATH))

    # my_upload(LOCAL_PATH, REMOTE_PATH, REMOTE_ROOT)




def loss_taobao(y_predict, y_true, exposure, y_index):

    y_exposure = 1/(1+exposure) * y_predict

    loss = (((y_exposure - y_true)**2) * (y_true + 1)).mean()

    return loss


if __name__ == '__main__':
    args = get_args()
    main(args)
