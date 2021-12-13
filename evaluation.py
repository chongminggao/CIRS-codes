# -*- coding: utf-8 -*-
# @Time    : 2021/7/26 5:01 下午
# @Author  : Chongming GAO
# @FileName: evaluation.py
import numpy as np
import torch



def test_kuaishou(model, env, dataset_val, is_softmax=True, epsilon=0, is_ucb=False):
    cumulative_reward = 0
    total_click_loss = 0
    total_turns = 0
    num_trajectory = 100

    for i in range(num_trajectory):
        user = env.reset()
        real_user_id = env.lbe_user.inverse_transform(user)

        done = False
        while not done:
            recommendation, reward_pred = model.recommend_k_item(real_user_id[0], dataset_val, k=1, is_softmax=is_softmax, epsilon=epsilon, is_ucb=is_ucb)

            # if need_transform:
            rec_small = env.lbe_photo.transform([recommendation])[0]

            state, reward, done, info = env.step(rec_small)

            total_turns += 1

            # metric 1
            cumulative_reward += reward

            # metric 2
            click_loss = np.absolute(reward_pred - reward)
            total_click_loss += click_loss

            if done:
                break

    ctr = cumulative_reward / total_turns
    click_loss = total_click_loss / total_turns

    # print('CTR: %.2f'.format(ctr))
    eval_result_RL = {"CTR": ctr, "click_loss": click_loss, "trajectory_len": total_turns / num_trajectory,
                      "trajectory_reward": cumulative_reward / num_trajectory}
    # if is_ucb:
    #     eval_result_RL.update({"ucb_n": model.n_each})

    return eval_result_RL



def test_taobao(model, env, epsilon=0):
# test the model in the interactive system
    cumulative_reward = 0
    total_click_loss = 0
    total_turns = 0
    num_trajectory = 100

    for i in range(num_trajectory):
        features = env.reset()
        done = False
        while not done:
            res = model(torch.FloatTensor(features).to(model.device).unsqueeze(0)).to('cpu').squeeze()
            item_feat_predict = res[model.y_index['feat_item'][0]:model.y_index['feat_item'][1]]
            action = item_feat_predict.detach().numpy()

            if epsilon > 0 and np.random.random() < epsilon:
                # Activate epsilon greedy
                action = np.random.random(action.shape)

            reward_pred = res[model.y_index['y'][0]:model.y_index['y'][1]]

            features, reward, done, info = env.step(action)

            total_turns += 1

            # metric 1
            cumulative_reward += reward

            # metric 2
            click_loss = np.absolute(float(reward_pred.detach().numpy()) - reward)
            total_click_loss += click_loss

            if done:
                break

    ctr = cumulative_reward / total_turns # /10
    click_loss = total_click_loss / total_turns

    # print('CTR: %.2f'.format(ctr))
    eval_result_RL = {"CTR": ctr,
                      "click_loss": click_loss,
                      "trajectory_len":total_turns/num_trajectory,
                      "trajectory_reward": cumulative_reward/num_trajectory} #/10}


    return eval_result_RL
