# -*- coding: utf-8 -*-
import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from CPC.warm_up import warm_up_cpc
from CPC.Encoder import ResEncoder
from Agent import Agent
from Tool.Actions import restart, ReleaseAll, take_action
from Tool.GetHP import Hp_getter
from Tool.FrameBuffer import FrameBuffer
from CPC.tools import Buffer, SimSiam

K_epochs = 80  # update policy for K epochs in one PPO update
eps_clip = 0.2  # clip parameter for PPO
gamma = 0.99  # discount factor

warm_up_epoch = 100
warm_up_episode = 500

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
        return -50
    if getter.get_boss_hp() < boss_hp:
        return 50
    return 1


def run_episode(getter, agent, buffer=None):
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

    step = 0
    done = 0
    episode_reward = 0
    start_time = time.time()
    while not done:
        hp = getter.get_self_hp()
        boss_hp = getter.get_boss_hp()
        obs = framebuffer.get_frame()
        action = agent.sample_action(obs)
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
        if step % 10 == 0 and buffer is not None:
            buffer.append(obs)
    return episode_reward, step, done


if __name__ == '__main__':
    # paused = True
    # paused = Tool.Helper.pause_game(paused)
    framebuffer = FrameBuffer()
    getter = Hp_getter()

    if os.path.exists(os.path.join('model', 'simsiam_best.pkl')):
        simsiam = torch.load(os.path.join('model', 'simsiam_best.pkl'))
        encoder = simsiam.encoder
        print('Load model successfully.')
        train_cpc = input("Whether to train CPC?")
        if not train_cpc:
            warm_up_epoch = 0
    else:
        encoder = ResEncoder(in_channels=1, out_dims=state_dim)
        simsiam = SimSiam(encoder, state_dim, pred_dim)

    # encoder = torch.load(os.path.join('model', 'encoder_best.pkl'))

    simsiam = simsiam.cuda()
    encoder = encoder.cuda()
    img_buffer = Buffer(_max_replay_buffer_size=50)
    agent = Agent(encoder, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std)
    writer = SummaryWriter()
    init_point = getter.get_play_location()

    epoch = 0
    best_loss = 999
    if warm_up_epoch:
        print("Start Collecting Samples for CPC")
        while epoch < warm_up_epoch:
            run_episode(getter, agent, buffer=img_buffer)
            loss = warm_up_cpc(simsiam, img_buffer,
                               epoch=epoch, warm_up_episode=warm_up_episode, writer=writer)
            print('CPC Epoch %s: loss = %s' % (epoch, loss))
            epoch += 1
            if loss < best_loss:
                best_loss = loss
                torch.save(simsiam, os.path.join('model', 'simsiam_best.pkl'))
                torch.save(encoder, os.path.join('model', 'encoder_best.pkl'))
                print('Best loss update: ', loss)
        print("Warmed up CPC for %s episodes" % str(epoch * warm_up_episode))
        del img_buffer

    print("Start training PPO")
    episode = 0
    while episode < 30000:
        # print('start one episode')
        episode += 1
        total_reward, total_step, done = run_episode(getter, agent)
        writer.add_scalar('Hornet' + ' /' + 'Reward', total_reward, episode)
        writer.add_scalar('Hornet' + ' /' + 'Win', max(0, done), episode)
        print("Episode: ", episode, ", Reward: ", total_reward, end=", ")
        print('Win!') if done == 1 else print()
        if episode % 3 == 0:
            agent.update()
        time.sleep(3.2)
