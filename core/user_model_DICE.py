# -*- coding: utf-8 -*-
# @Time    : 2021/7/27 3:31 下午
# @Author  : Chongming GAO
# @FileName: user_model_mmoe.py

import torch
from deepctr_torch.inputs import combined_dnn_input, build_input_features
from deepctr_torch.layers import DNN, PredictionLayer, FM
from torch import nn

from core.layers import Linear
from core.user_model import UserModel, compute_input_dim


class UserModel_DICE(UserModel):
    """Instantiates the Multi-gate Mixture-of-Experts architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param num_tasks: integer, number of tasks, equal to number of outputs, must be greater than 1.
    :param tasks: list of str, indicating the loss of each tasks, ``"binary"`` for  binary logloss, ``"regression"`` for regression loss. e.g. ['binary', 'regression']
    :param num_experts: integer, number of experts.
    :param expert_dim: integer, the hidden units of each expert.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of shared-bottom DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param task_dnn_units: list,list of positive integer or empty list, the layer number and units in each layer of task-specific DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param device: str, ``"cpu"`` or ``"cuda:0"``

    :return: A PyTorch model instance.
    """

    def __init__(self, feature_columns, y_columns, task, task_logit_dim,
                 dnn_hidden_units=(128, 128),
                 l2_reg_embedding=1e-5, l2_reg_dnn=1e-1, init_std=0.0001, task_dnn_units=None, seed=2021, dnn_dropout=0,
                 dnn_activation='relu', dnn_use_bn=False, device='cpu', padding_idx=None):

        super(UserModel_DICE, self).__init__(feature_columns, y_columns,
                                             l2_reg_embedding=l2_reg_embedding,
                                             init_std=init_std, seed=seed, device=device,
                                             padding_idx=padding_idx)

        self.feature_columns = feature_columns
        self.y_columns = y_columns
        self.task_logit_dim = task_logit_dim

        self.sigmoid = nn.Sigmoid()
        """
        For MMOE Layer
        """
        self.task = task
        self.task_dnn_units = task_dnn_units

        # prepare feature columns and feature index

        self.feature_main = self.feature_columns[:9]
        self.feature_ui_int = [self.feature_columns[0]] + [self.feature_columns[2]]
        self.feature_ui_con = [self.feature_columns[1]] + [self.feature_columns[3]]
        # self.feature_index_main = OrderedDict({k: v for i, (k, v) in enumerate(self.feature_index.items()) if i < 9})
        # self.feature_index_ui = OrderedDict({k: v for i, (k, v) in enumerate(self.feature_index.items()) if i in [1, 3]})
        self.index_main = build_input_features(self.feature_main)
        self.index_ui_int = build_input_features(self.feature_ui_int)
        self.index_ui_con = build_input_features(self.feature_ui_con)

        # set networks

        self.dnn_main = DNN(compute_input_dim(self.feature_main), dnn_hidden_units,
                            activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                            init_std=init_std, device=device)
        self.last_main = nn.Linear(dnn_hidden_units[-1], 1, bias=False)
        self.out_main = PredictionLayer(task, 1)

        self.dnn_ui = DNN(compute_input_dim(self.feature_ui_int), dnn_hidden_units,
                          activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                          init_std=init_std, device=device)
        self.last_ui = nn.Linear(dnn_hidden_units[-1], 1, bias=False)
        self.out_ui = PredictionLayer(task, 1)

        """
        For DeepFM Layer.
        """
        use_fm = True if task_logit_dim == 1 else False
        self.use_fm = use_fm

        self.fm_task = FM() if use_fm else None

        self.linear_main = Linear(self.feature_main, self.index_main, device=device)
        self.linear_ui = Linear(self.feature_ui_int, self.index_ui_int, device=device)

        self.add_regularization_weight(self.parameters(), l2=l2_reg_dnn)


        self.to(device)

    # def _mmoe(self, X, feature_columns, is_sigmoid):
    #     sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, feature_columns,
    #                                                                               self.embedding_dict)
    #
    #     dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
    #     dnn_output = self.dnn(dnn_input)
    #     mmoe_outs = self.mmoe_layer(dnn_output)
    #     if self.task_dnn_units is not None:
    #         mmoe_outs = [self.task_dnn[i](mmoe_out) for i, mmoe_out in enumerate(mmoe_outs)]
    #
    #     task_outputs = []
    #     for i, mmoe_out in enumerate(mmoe_outs):
    #         logit = self.tower_network[i](mmoe_out)
    #
    #         if is_sigmoid:
    #             output = self.out[i](logit)
    #             task_outputs.append(output)
    #         else:
    #             task_outputs.append(logit)
    #
    #     task_outputs = torch.cat(task_outputs, -1)
    #     return task_outputs

    def _deepfm(self, X, feature_columns, feature_index, score=None, is_main=True):

        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, feature_columns,
                                                                                  self.embedding_dict,
                                                                                  feature_index=feature_index)
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

        if is_main:
            linear_model = self.linear_main
            dnn = self.dnn_main
            last = self.last_main
            out = self.out_main
        else:
            linear_model = self.linear_ui
            dnn = self.dnn_ui
            last = self.last_ui
            out = self.out_ui

        # Linear and FM logit
        logit = torch.zeros([len(X), self.task_logit_dim], device=X.device)

        if linear_model is not None:
            logit = logit + linear_model(X)

            fm_model = self.fm_task
            if self.use_fm and len(sparse_embedding_list) > 0 and fm_model is not None:
                fm_input = torch.cat(sparse_embedding_list, dim=1)
                logit += fm_model(fm_input)

        linear_logit = logit

        # DNN
        dnn_logit = out(last(dnn(dnn_input)))

        y_pred = linear_logit + dnn_logit

        y_score = y_pred * score if score is not None else y_pred

        return y_score

    def get_loss(self, x, y, score):

        X_pos = x[:, :9]
        X_neg = torch.cat([x[:, :2], x[:, 9:]], dim=1)




        y_deepfm_pos = self._deepfm(X_pos, self.feature_main, self.index_main, is_main=True)
        y_deepfm_neg = self._deepfm(X_neg, self.feature_main, self.index_main, is_main=True)

        X_pos_int = x[:, [0, 2]]
        X_neg_int = x[:, [0, 9]]
        X_pos_con = x[:, [1, 3]]
        X_neg_con = x[:, [1, 10]]

        y_deepfm_pos_int = self._deepfm(X_pos_int, self.feature_ui_int, self.index_ui_int, is_main=False)
        y_deepfm_neg_int = self._deepfm(X_neg_int, self.feature_ui_int, self.index_ui_int, is_main=False)
        y_deepfm_pos_con = self._deepfm(X_pos_con, self.feature_ui_con, self.index_ui_con, is_main=False)
        y_deepfm_neg_con = self._deepfm(X_neg_con, self.feature_ui_con, self.index_ui_con, is_main=False)

        loss = self.loss_func(y, y_deepfm_pos, y_deepfm_neg,
                              y_deepfm_pos_int, y_deepfm_neg_int,
                              y_deepfm_pos_con, y_deepfm_neg_con, score)

        return loss

    def forward(self, x, score=None):
        x2 = torch.cat([x[:,0:1], x[:,0:2], x[:,1:]], dim=-1)
        y_deepfm = self._deepfm(x2, self.feature_main, self.index_main, score=score, is_main=True)
        return y_deepfm
