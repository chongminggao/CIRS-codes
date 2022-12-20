# -*- coding: utf-8 -*-
# @Time    : 2021/10/1 3:03 下午
# @Author  : Chongming GAO
# @FileName: kuaishouEnv.py
import json
import os
import pickle

import gym
import torch
from gym import spaces
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from collections import Counter
import itertools

import pandas as pd
import numpy as np
import random

from tqdm import tqdm

from core.util import get_distance_mat

CODEPATH = os.path.dirname(__file__)
ROOTPATH = os.path.dirname(CODEPATH)
DATAPATH = os.path.join(ROOTPATH, "data")


class KuaishouEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, mat=None, lbe_user=None, lbe_photo=None, list_feat=None, df_photo_env=None, df_dist_small=None,
                 num_leave_compute=5, leave_threshold=1, max_turn=100):

        self.max_turn = max_turn

        if mat is not None:
            self.mat = mat
            self.lbe_user = lbe_user
            self.lbe_photo = lbe_photo
            self.list_feat = list_feat
            self.df_photo_env = df_photo_env
            self.df_dist_small = df_dist_small
        else:
            self.mat, self.lbe_user, self.lbe_photo, self.list_feat, self.df_photo_env, self.df_dist_small = self.load_mat()


        self.list_feat_small = list(map(lambda x: self.list_feat[x], self.lbe_photo.classes_))

        # smallmat shape: (1411, 3327)

        self.observation_space = spaces.Box(low=0, high=len(self.mat) - 1, shape=(1,), dtype=np.int32)
        self.action_space = spaces.Box(low=0, high=self.mat.shape[1] - 1, shape=(1,), dtype=np.int32)

        self.num_leave_compute = num_leave_compute
        self.leave_threshold = leave_threshold

        self.reset()

    @staticmethod
    def load_mat():
        small_path = os.path.join(DATAPATH, "small_matrix.csv")
        df_small = pd.read_csv(small_path, header=0, usecols=['user_id', 'photo_id', 'watch_ratio'])
        # df_small['watch_ratio'][df_small['watch_ratio'] > 5] = 5
        df_small.loc[df_small['watch_ratio'] > 5, 'watch_ratio'] = 5

        lbe_photo = LabelEncoder()
        lbe_photo.fit(df_small['photo_id'].unique())

        lbe_user = LabelEncoder()
        lbe_user.fit(df_small['user_id'].unique())

        mat = csr_matrix(
            (df_small['watch_ratio'],
             (lbe_user.transform(df_small['user_id']), lbe_photo.transform(df_small['photo_id']))),
            shape=(df_small['user_id'].nunique(), df_small['photo_id'].nunique())).toarray()

        mat[np.isnan(mat)] = df_small['watch_ratio'].mean()
        mat[np.isinf(mat)] = df_small['watch_ratio'].mean()

        # load categories:
        print("load item feature")
        filepath = os.path.join(DATAPATH, 'item_categories.json')
        with open(filepath, 'r') as file:
            data_feat = json.load(file)
        print("number of items:", len(data_feat))
        list_feat = [0] * len(data_feat)
        for i in range(len(data_feat)):
            list_feat[i] = data_feat[str(i)]['feature_index']

        df_feat = pd.DataFrame(list_feat, columns=['feat0', 'feat1', 'feat2', 'feat3'], dtype=int)
        df_feat.index.name = "photo_id"
        df_feat[df_feat.isna()] = -1
        df_feat = df_feat + 1
        df_feat = df_feat.astype(int)

        photo_mean_duration_path = os.path.join(DATAPATH, "photo_mean_duration.json")
        with open(photo_mean_duration_path, 'r') as file:
            photo_mean_duration = json.load(file)
        photo_mean_duration = {int(k): v for k, v in photo_mean_duration.items()}

        photo_list = df_small['photo_id'].unique()
        df_photo_env = df_feat.loc[photo_list]
        df_photo_env['photo_duration'] = np.array(
            list(map(lambda x: photo_mean_duration[x], df_photo_env.index)))

        # load or construct the distance mat (between item pairs):
        df_dist_small = get_distance_mat(list_feat, lbe_photo.classes_, DATAPATH=DATAPATH)

        return mat, lbe_user, lbe_photo, list_feat, df_photo_env, df_dist_small

    @staticmethod
    def compute_normed_reward(user_model, lbe_user, lbe_photo, df_photo_env):
        # filename = "normed_reward.pickle"
        # filepath = os.path.join(DATAPATH, filename)

        # if os.path.isfile(filepath):
        #     with open(filepath, "rb") as file:
        #         normed_mat = pickle.load(file)
        #     return normed_mat

        n_user = len(lbe_user.classes_)
        n_item = len(lbe_photo.classes_)

        item_info = df_photo_env.loc[lbe_photo.classes_]
        item_info["photo_id"] = item_info.index
        item_info = item_info[["photo_id", "feat0", "feat1", "feat2", "feat3", "photo_duration"]]
        item_np = item_info.to_numpy()

        predict_mat = np.zeros((n_user, n_item))

        for i, user in tqdm(enumerate(lbe_user.classes_), total=n_user, desc="predict all users' rewards on all items"):
            ui = torch.tensor(np.concatenate((np.ones((n_item, 1)) * user, item_np), axis=1),
                              dtype=torch.float, device=user_model.device, requires_grad=False)
            reward_u = user_model.forward(ui).detach().squeeze().cpu().numpy()
            predict_mat[i] = reward_u

        minn = predict_mat.min()
        maxx = predict_mat.max()


        normed_mat = (predict_mat - minn) / (maxx - minn)

        return normed_mat

    @property
    def state(self):
        if self.action is None:
            res = self.cur_user
        else:
            res = self.action
        return np.array([res])

    def __user_generator(self):
        user = random.randint(0, len(self.mat) - 1)
        # # todo for debug
        # user = 0
        return user

    def step(self, action):
        # action = int(action)

        # Action: tensor with shape (32, )
        self.action = action
        t = self.total_turn
        done = self._determine_whether_to_leave(t, action)
        if t >= (self.max_turn-1):
            done = True
        self._add_action_to_history(t, action)

        reward = self.mat[self.cur_user, action]

        self.cum_reward += reward
        self.total_turn += 1

        # if done:
        #     self.cur_user = self.__user_generator()

        return self.state, reward, done, {'cum_reward': self.cum_reward}

    def reset(self):
        self.cum_reward = 0
        self.total_turn = 0
        self.cur_user = self.__user_generator()

        self.action = None  # Add by Chongming
        self._reset_history()

        return self.state

    def render(self, mode='human', close=False):
        history_action = self.history_action
        category = {k:self.list_feat_small[v] for k,v in history_action.items()}
        # category_debug = {k:self.list_feat[v] for k,v in history_action.items()}
        # return history_action, category, category_debug
        return self.cur_user, history_action, category

    def _determine_whether_to_leave(self, t, action):
        # self.list_feat[action]
        if t == 0:
            return False
        window_actions = self.sequence_action[t - self.num_leave_compute:t]
        hist_categories_each = list(map(lambda x: self.list_feat_small[x], window_actions))

        # hist_set = set.union(*list(map(lambda x: self.list_feat[x], self.sequence_action[t - self.num_leave_compute:t-1])))

        hist_categories = list(itertools.chain(*hist_categories_each))
        hist_dict = Counter(hist_categories)
        category_a = self.list_feat_small[action]
        for c in category_a:
            if hist_dict[c] > self.leave_threshold:
                return True

        # if action in window_actions:
        #     return True

        return False

    def _reset_history(self):
        self.history_action = {}
        self.sequence_action = []
        self.max_history = 0

    def _add_action_to_history(self, t, action):

        self.sequence_action.append(action)
        self.history_action[t] = action

        assert self.max_history == t
        self.max_history += 1

