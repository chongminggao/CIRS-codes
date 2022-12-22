# -*- coding: utf-8 -*-
# @Time    : 2021/8/2 4:04 下午
# @Author  : Chongming GAO
# @FileName: simulated_env.py
from collections import defaultdict

import gym

from torch import FloatTensor
from tqdm import tqdm

from core.util import compute_action_distance, clip0, compute_exposure
from virtualTB.model.UserModel import UserModel
from virtualTB.utils import *


class SimulatedEnv(gym.Env):

    def __init__(self, user_model: UserModel, task_name: str = "VirtualTB-v0", version: str = "v1", tau: float = 1.0,
                 use_exposure_intervention=True,
                 alpha_u=None, beta_i=None,
                 normed_mat=None,
                 gamma_exposure=1, r_decay=1):
        self.user_model = user_model.eval()

        self.env_task = gym.make(task_name)

        self.observation_space = self.env_task.observation_space
        self.action_space = self.env_task.action_space
        self.cum_reward = 0  # total_a in virtualtaobao
        self.total_turn = 0  # total_c in virtualtaobao
        self.env_name = task_name
        self.version = version
        self.tau = tau
        self.use_exposure_intervention = use_exposure_intervention
        self.alpha_u = alpha_u
        self.beta_i = beta_i
        self.normed_mat = normed_mat
        self.gamma_exposure = gamma_exposure
        self.r_decay = r_decay

        self._reset_history()

    # def compile(self, num_env=1):
    #     self.env_list = DummyVectorEnv([lambda: gym.make(self.env_task) for _ in range(num_env)])

    def _construct_state(self, reward):

        if self.env_name == "VirtualTB-v0":
            res = np.concatenate((self.action, np.array([reward, 0.0, self.total_turn])), axis=-1)  # [29,9,100]
        elif self.env_name == "KuaishouEnv-v0":
            res = self.env_task.state

        return res

    def seed(self, sd=0):
        torch.manual_seed(sd)

    def reset(self):
        self.cum_reward = 0
        self.total_turn = 0
        self.reward = 0
        self.action = None
        self.env_task.action = None
        self.state = self.env_task.reset()

        self._reset_history()
        if self.env_name == "VirtualTB-v0":
            self.cur_user = self.state[:-3]
        elif self.env_name == "KuaishouEnv-v0":
            self.cur_user = self.state
        return self.state

    def render(self, mode='human', close=False):
        self.env_task.render(mode)

    def _compute_pred_reward(self, exposure_effect, action):
        if self.env_name == "VirtualTB-v0":
            feature = np.concatenate((self.cur_user, np.array([self.reward, 0, self.total_turn]), action), axis=-1)
            feature_tensor = torch.unsqueeze(torch.tensor(feature, device=self.user_model.device, dtype=torch.float), 0)
            # pred_reward = self.user_model(feature_tensor).detach().cpu().numpy().squeeze().round()
            pred_reward = self.user_model.forward(feature_tensor).detach().cpu().numpy().squeeze()
            if pred_reward < 0:
                pred_reward = 0
            if pred_reward > 10:
                pred_reward = 10
        elif self.env_name == "KuaishouEnv-v0":
            # real_user = self.env_task.lbe_user.inverse_transform(self.cur_user)
            # user = np.expand_dims(real_user, 0)
            #
            # df_photo_env = self.env_task.df_photo_env
            # item_index = self.env_task.lbe_photo.inverse_transform([action])
            # item_info = df_photo_env.loc[item_index].to_numpy()
            #
            # u_i = torch.tensor(np.concatenate((user, np.expand_dims(item_index, 0), item_info), axis=1),
            #                    dtype=torch.float, device=self.user_model.device, requires_grad=False)
            #
            # pred_reward = self.user_model.forward(u_i).detach().squeeze().cpu().numpy()

            pred_reward = self.normed_mat[self.cur_user[0], action]

        if self.version == "v1":
            # version 1
            final_reward = clip0(pred_reward) / (1.0 + exposure_effect)
        else:
            # version 2
            final_reward = clip0(pred_reward - exposure_effect)

        return final_reward

    def step(self, action: FloatTensor):
        # 1. Collect ground-truth transition info
        self.action = action
        real_state, real_reward, real_done, real_info = self.env_task.step(action)

        # 2. Compute intervened exposure effect e^*_t(u, i)
        t = int(self.total_turn)
        if self.use_exposure_intervention:
            exposure_effect = self._compute_exposure_effect(t, action)
        else:
            exposure_effect = 0

        if t < self.env_task.max_turn:
            self._add_action_to_history(t, action, exposure_effect)

        # 3. Predict click score, i.e, reward
        pred_reward = self._compute_pred_reward(exposure_effect, action)

        if self.env_name == "KuaishouEnv-v0":
            num_repeat = self.num_actions[action] - 1 # minus itself
            decay = self.r_decay ** num_repeat
            pred_reward = pred_reward * decay

        self.reward = pred_reward
        self.cum_reward += pred_reward
        self.total_turn = self.env_task.total_turn

        done = real_done
        # Rethink commented, do not use new user as new state
        # if done:
        #     self.state = self.env_task.reset()

        self.state = self._construct_state(pred_reward)

        return self.state, pred_reward, done, {'CTR': self.cum_reward / self.total_turn / 10}

    def _compute_exposure_effect(self, t, action):

        if t == 0:
            return 0

        a_history = self.history_action[:t]
        distance = compute_action_distance(action, a_history, self.env_name, self.env_task)
        t_diff = t - np.arange(t)
        exposure_effect = compute_exposure(t_diff, distance, self.tau)

        if self.alpha_u is not None:
            u_id = self.env_task.lbe_user.inverse_transform(self.cur_user)[0]
            p_id = self.env_task.lbe_photo.inverse_transform([action])[0]
            a_u = self.alpha_u[u_id]
            b_i = self.beta_i[p_id]
            exposure_effect_new = float(exposure_effect * a_u * b_i)
        else:
            exposure_effect_new = exposure_effect

        exposure_gamma = exposure_effect_new * self.gamma_exposure

        return exposure_gamma

    def _reset_history(self):
        # self.history_action = {}
        if self.env_name == "VirtualTB-v0":
            # self.history_action = np.empty([0, self.action_space.shape[0]])
            self.history_action = np.zeros([self.env_task.max_turn, self.env_task.action_space.shape[0]])
        elif self.env_name == "KuaishouEnv-v0":
            self.history_action = np.zeros(self.env_task.max_turn, dtype=np.int)
        self.history_exposure = {}
        self.num_actions = defaultdict(int)
        self.max_history = 0

    def _add_action_to_history(self, t, action, exposure):
        if self.env_name == "VirtualTB-v0":
            action2 = np.expand_dims(action, 0)
            # self.history_action = np.append(self.history_action, action2, axis=0)
            self.history_action[t] = action2
        elif self.env_name == "KuaishouEnv-v0":
            self.history_action[t] = action
            self.num_actions[action] += 1

        self.history_exposure[t] = exposure

        assert self.max_history == t
        self.max_history += 1
