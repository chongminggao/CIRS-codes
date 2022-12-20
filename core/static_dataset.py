# -*- coding: utf-8 -*-
# @Author  : Chongming GAO
# @FileName: static_dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset


class StaticDataset(Dataset):
    def __init__(self, x_columns, y_columns, num_workers=4):
        self.x_columns = x_columns
        self.y_columns = y_columns

        self.num_workers = num_workers

        self.len = 0
        self.neg_items_info = None

    def set_env_items(self, df_small, df_feat, photo_mean_duration):  # for kuaishou data
        photo_list = df_small['photo_id'].unique()
        self.df_photo_env = df_feat.loc[photo_list]
        self.df_photo_env['photo_duration'] = np.array(
            list(map(lambda x: photo_mean_duration[x], self.df_photo_env.index)))

    def compile_dataset(self, df_x, df_y, score=None):
        self.x_numpy = df_x.to_numpy()
        self.y_numpy = df_y.to_numpy()

        if score is None:
            self.score = np.zeros([len(self.x_numpy), 1])
        else:
            self.score = score

        self.len = len(self.x_numpy)

    def get_dataset_train(self):
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(self.x_numpy),
                                                 torch.from_numpy(self.y_numpy),
                                                 torch.from_numpy(self.score))
        return dataset

    def get_dataset_eval(self):
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(self.x_numpy),
                                                 torch.from_numpy(self.y_numpy))
        return dataset

    def get_y(self):
        return self.y_numpy

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index: int):
        x = self.x_numpy[index]
        y = self.y_numpy[index]
        return x, y
