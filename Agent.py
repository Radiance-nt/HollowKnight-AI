# -*- coding: utf-8 -*-
import os
import random
from PPO import PPO, RolloutBuffer
import numpy as np
import torch
from torch import nn
from torch.distributions import MultivariateNormal, Categorical

device = torch.device('cuda:0') if (torch.cuda.is_available()) else torch.device('cpu')


class Agent:
    def __init__(self, encoder, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std):
        self.encoder = encoder
        self.buffer = RolloutBuffer()
        self.algorithm = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                             action_std_init=action_std, buffer=self.buffer)

    def sample_action(self, obs):
        if not isinstance(obs, torch.Tensor):
            if len(obs.shape) < 3:
                obs = torch.Tensor(obs).unsqueeze(0)
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
        self.algorithm.update()

    def save(self, path='model', episode=0):
        path = os.path.join(path, 'encoder_' + str(episode) + '.pkl')
        self.encoder._save_to_state_dict(path)
        self.algorithm.save(path, episode)
