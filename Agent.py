# -*- coding: utf-8 -*-
import os
import random
import time

from PPO import PPO, RolloutBuffer
import numpy as np
import torch
from torch import nn
from torch.distributions import MultivariateNormal, Categorical

device = torch.device('cuda:0') if (torch.cuda.is_available()) else torch.device('cpu')


class Agent:
    def __init__(self, encoder, stack_num, state_dim, action_dim,
                 lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std):
        self.encoder = encoder
        self.buffer = RolloutBuffer()
        self.algorithm = PPO(stack_num, state_dim, action_dim,
                             lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                             action_std_init=action_std, buffer=self.buffer)

    def sample_action(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.Tensor(obs)
            if len(obs.shape) < 4:
                obs = obs.unsqueeze(0)
        obs = obs.cuda()
        with torch.no_grad():
            code = self.encoder(obs)
            out = self.algorithm.select_action(code)

        dic = {'lr': out[0], 'ud': out[1], 'Z': out[2], 'X': out[3], 'C': out[4], 'F': out[5]}

        return dic

    def random_sample_action(self, obs, hp):
        lr, ud, z, x, c, f = [random.random() for i in range(6)]
        dic = {'lr': lr, 'ud': ud, 'Z': z, 'X': x, 'C': c, 'F': f, }
        return dic

    def update(self):
        infos = self.algorithm.update()
        return infos

    def save(self, path='model', episode=0):
        path = os.path.join(path, 'encoder_' + str(episode) + '.pkl')
        self.encoder._save_to_state_dict(path)
        self.algorithm.save(path, episode)


def agent_update_process(agent, training_rl_episode, writer):
    episode = 0
    while episode < training_rl_episode:
        if len(agent.buffer) > 400:
            infos = agent.update()
            log(infos, episode, writer)
            # if episode % 10 == 0:
            print('Training PPO in %s Epoch.' % episode)
            if episode % 50 == 0:
                agent.algorithm.save('model', episode)
            episode += 1
        else:
            time.sleep(5)


def log(infos, episode, writer):
    for item in infos:
        writer.add_scalar('PPO' + ' /' + item, infos[item], episode)
