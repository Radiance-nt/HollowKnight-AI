# -*- coding: utf-8 -*-
import os
import time
from threading import Thread

import torch
from torch.utils.tensorboard import SummaryWriter

from CPC.warm_up import warm_up_process
from CPC.Encoder import ResEncoder
from Agent import Agent, agent_update_process
from Tool.Actions import restart, ReleaseAll, take_action
from Tool.GetHP import Hp_getter
from Tool.FrameGetter import FrameGetter
from CPC.tools import Buffer, SimSiam

colormode = 3
stack_num = 4
stack_stride = 4
K_epochs = 5  # update policy for K epochs in one PPO update
eps_clip = 0.2  # clip parameter for PPO
gamma = 0.7  # discount factor

warm_up_epoch = 25
warm_up_episode = 400
training_rl_episode = 10000

lr_actor = 0.0003  # learning rate for actor network
lr_critic = 0.001  # learning rate for critic network
action_std = 0.6  # starting std for action distribution (Multivariate Normal)
action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)
action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)

dim = 256
pred_dim = 64
state_dim = dim
action_dim = 5


def cal_reward(getter, hp, boss_hp):
    if getter.get_self_hp() < hp:
        return -800
    if getter.get_boss_hp() < boss_hp:
        return 80
    return 1


def run_episode(getter, agent, obs_buffer, img_buffer=None):
    step = 0
    restart()
    while True:
        current = getter.get_play_location()
        step += 1
        if (current[0] - init_point[0]) ** 2 + (current[1] - init_point[1]) ** 2 > 16:
            break
        if step >= 20:
            step = 0
            restart()
        time.sleep(0.5)
    obs_buffer.clear()
    step = 0
    done = 0
    episode_reward = 0
    while not done:
        hp = getter.get_self_hp()
        boss_hp = getter.get_boss_hp()
        obs = framegetter.get_frame()
        obs_buffer.append(obs)
        stack_obs = obs_buffer.get_stack(length=stack_num, stride=stack_stride)
        # print('stack_obs.shape',stack_obs.shape)
        action = agent.sample_action(stack_obs)
        take_action(action)
        reward = cal_reward(getter, hp, boss_hp)
        agent.buffer.rewards.append(reward)
        agent.buffer.is_terminals.append(done)
        current = getter.get_play_location()
        if hp == 0:
            done = -1
            ReleaseAll()
        if boss_hp > 900 and hp > 0 and episode_reward > 50 and \
                (current[0] - init_point[0]) ** 2 + (current[1] - init_point[1]) ** 2 < 16:
            done = 1
            ReleaseAll()
        episode_reward += reward
        step += 1
        if step % 1 == 0 and img_buffer is not None:
            img_buffer.append(obs)
    return episode_reward, step, done


if __name__ == '__main__':
    assert colormode == 1 or colormode == 4 or colormode == 3
    framegetter = FrameGetter(colormode)
    getter = Hp_getter()
    cpc_model_name = os.path.join('model', 'encoder', 'simsiam_' + str(colormode) + 'channel_')
    if os.path.exists(cpc_model_name):
        simsiam = torch.load(cpc_model_name + 'best.pkl')
        encoder = simsiam.encoder
        print('Loading Encoder Network successfully.')
        train_cpc = input("Whether to train CPC?")
        if not train_cpc:
            warm_up_epoch = 0
    else:
        encoder = ResEncoder(in_channels=colormode, out_dims=state_dim)
        simsiam = SimSiam(encoder, state_dim, pred_dim)
        print('Create Encoder successfully.')

    simsiam = simsiam.cuda()
    encoder = encoder.cuda()
    obs_buffer = Buffer(_length=stack_num, _stride=stack_stride, _max_replay_buffer_size=20)
    img_buffer = Buffer(_length=1, _max_replay_buffer_size=800)
    writer = SummaryWriter()

    agent = Agent(encoder, stack_num, state_dim, action_dim,
                  lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std)

    # agent.algorithm.load(os.path.join('model', 'PPO_best.pkl'))

    init_point = getter.get_play_location()
    warm_process = None
    training_rl_process = None
    episode = 0

    print('Training CPC = ', bool(warm_up_epoch))
    if warm_up_epoch:
        print('Warm up epoch =', warm_up_epoch, end='; ')
        print('Episodes for each epoch =', warm_up_episode)

    while episode < 30000:
        total_reward, total_step, done = run_episode(getter, agent, obs_buffer=obs_buffer, img_buffer=img_buffer)
        writer.add_scalar('Hornet' + ' /' + 'Reward', total_reward, episode)
        writer.add_scalar('Hornet' + ' /' + 'Win', max(0, done), episode)
        print("Episode: ", episode, ", Reward: ", total_reward, end=", ")
        print('Win!') if done == 1 else print()

        if warm_process is None:
            warm_process = Thread(target=warm_up_process, args=(
                simsiam, img_buffer, warm_up_epoch, warm_up_episode, writer, cpc_model_name))
            warm_process.start()

        if training_rl_process is None:
            training_rl_process = Thread(target=agent_update_process, args=(agent, training_rl_episode, writer))
            training_rl_process.start()

        time.sleep(3.5)
        episode += 1
