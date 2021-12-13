import gym
from gym import spaces

from core.util import compute_action_distance
from virtualTB.model.ActionModel import ActionModel
from virtualTB.model.LeaveModel import LeaveModel
from virtualTB.model.UserModel import UserModel
from virtualTB.utils import *


class VirtualTB(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_leave_compute=5, leave_threshold=4.5, max_turn=100):
        # self.n_item = 5
        self.n_user_feature = 88  # categorical features
        self.n_item_feature = 27  # continue features
        self.max_turn = max_turn

        self.obs_low = np.concatenate(([0] * self.n_user_feature, [0, 0, 0]))
        self.obs_high = np.concatenate(([1] * self.n_user_feature, [29, 9, 100]))
        self.observation_space = spaces.Box(low=self.obs_low, high=self.obs_high, dtype=np.int32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_item_feature,), dtype=np.float32)
        self.user_model = UserModel()
        self.user_model.load()
        self.user_action_model = ActionModel()
        self.user_action_model.load()
        self.user_leave_model = LeaveModel()
        self.user_leave_model.load()
        self.static = False

        self.num_leave_compute = num_leave_compute
        self.leave_threshold = leave_threshold

        self.reset()

    def set_state_mode(self, is_static=False):
        self.static = is_static

    def seed(self, sd=0):
        torch.manual_seed(sd)

    # @property
    # def state(self):
    #     return np.concatenate((self.cur_user, self.lst_action, np.array([self.total_c])), axis = -1) # [29,9,100]

    @property
    def state(self):
        if self.static:
            res = np.concatenate((self.cur_user, self.lst_action, np.array([self.total_turn])), axis=-1)  # [29,9,100]
            return res

        # Revised by Chongming:
        if self.action is None:
            res = np.concatenate((self.cur_user, self.lst_action, np.array([self.total_turn])), axis=-1)  # [29,9,100]
        else:
            res = np.concatenate((self.action, self.lst_action, np.array([self.total_turn])), axis=-1)  # [29,9,100]

        # res = np.concatenate((self.cur_user, self.lst_action, np.array([self.total_c])), axis=-1)  # [29,9,100]
        return res

        # return np.concatenate((self.cur_user, self.lst_action, np.array([self.total_c])), axis=-1)  # [29,9,100]

    def __user_generator(self):
        # with shape(n_user_feature,)
        user = self.user_model.generate()

        self.__leave = self.user_leave_model.predict(user)
        # self.__leave = 100

        return user


    def step(self, action):
        # Action: tensor with shape (27, )
        self.action = action
        t = self.total_turn
        done = self._determine_whether_to_leave(t, action)
        if t >= (self.max_turn-1):
            done = True

        self._add_action_to_history(t, action)

        self.lst_action = self.user_action_model.predict(FLOAT(self.cur_user).unsqueeze(0), FLOAT([[self.total_turn]]),
                                                         FLOAT(action).unsqueeze(0)).detach().numpy()[0]
        reward = int(self.lst_action[0])




        self.cum_reward += reward
        self.total_turn += 1
        self.rend_action = deepcopy(self.lst_action)


        if done:
            self.cur_user = self.__user_generator().squeeze().detach().numpy()
            self.lst_action = FLOAT([0, 0])

        return self.state, reward, done, {'CTR': self.cum_reward / self.total_turn / 10}

    def reset(self):
        self.cum_reward = 0
        self.total_turn = 0
        self.cur_user = self.__user_generator().squeeze().detach().numpy()
        self.lst_action = FLOAT([0, 0])
        self.rend_action = deepcopy(self.lst_action)

        self.action = None  # Add by Chongming
        self._reset_history()


        return self.state

    def render(self, mode='human', close=False):
        print('Current State:')
        print('\t', self.state)
        a, b = np.clip(self.rend_action, a_min=0, a_max=None)
        print('User\'s action:')
        print('\tclick:%2d, leave:%s, index:%2d' % (
            int(a), 'True' if self.total_turn > (self.max_turn-1) else 'False', int(self.total_turn)))
        print('Total clicks:', self.cum_reward)



    def _determine_whether_to_leave(self, t, action):
        for t_l in range(t-1, max(-1, t-self.num_leave_compute), -1):
            action_l = self.history_action[t_l]
            distance_l = compute_action_distance(action, action_l)

            if distance_l <= self.leave_threshold:
                return True
        return False


    def _reset_history(self):
        self.history_action = {}
        self.max_history = 0

    def _add_action_to_history(self, t, action):
        self.history_action[t] = action

        assert self.max_history == t
        self.max_history += 1