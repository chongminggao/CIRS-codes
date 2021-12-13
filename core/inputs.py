# -*- coding: utf-8 -*-
# @Time    : 2021/10/6 9:58 下午
# @Author  : Chongming GAO
# @FileName: inputs.py

# from collections import namedtuple
from deepctr_torch.inputs import SparseFeat, DenseFeat

DEFAULT_GROUP_NAME = "default_group"


class SparseFeatP(SparseFeat):
    def __new__(cls, name, vocabulary_size, embedding_dim=4, use_hash=False, dtype="int32", embedding_name=None,
                group_name=DEFAULT_GROUP_NAME, padding_idx=None):
        return super(SparseFeatP, cls).__new__(cls, name, vocabulary_size, embedding_dim, use_hash, dtype,
                                               embedding_name, group_name)

    def __init__(self, name, vocabulary_size, embedding_dim=4, use_hash=False, dtype="int32", embedding_name=None,
                group_name=DEFAULT_GROUP_NAME, padding_idx=None):
        self.padding_idx = padding_idx



def get_dataset_columns(dim_model, envname="VirtualTB-v0", env=None):
    user_columns, action_columns, feedback_columns = [], [], []
    has_user_embedding, has_action_embedding, has_feedback_embedding = None, None, None
    if envname == "VirtualTB-v0":
        user_columns = [DenseFeat("feat_user", 88)]
        action_columns = [DenseFeat("feat_item", 27)]
        # feedback_columns = [SparseFeat("feat_feedback", 11, embedding_dim=27)]
        feedback_columns = [DenseFeat("feat_feedback", 1)]
        has_user_embedding = True
        has_action_embedding = True
        has_feedback_embedding = True
    elif envname == "KuaishouEnv-v0":
        user_columns = [SparseFeatP("feat_user", env.mat.shape[0], embedding_dim=dim_model)]
        action_columns = [SparseFeatP("feat_item", env.mat.shape[1], embedding_dim=dim_model)]
        feedback_columns = [DenseFeat("feat_feedback", 1)]
        has_user_embedding = False
        has_action_embedding = False
        has_feedback_embedding = True

    return user_columns, action_columns, feedback_columns, \
           has_user_embedding, has_action_embedding, has_feedback_embedding
