# -*- coding: utf-8 -*-
import os
import random
import time

import cv2
from torch.nn import GroupNorm

from PPO import PPO, RolloutBuffer
import numpy as np
import torch
from torch import nn
from torch.distributions import MultivariateNormal, Categorical

device = torch.device('cuda:0') if (torch.cuda.is_available()) else torch.device('cpu')

forward_maps = []


def forward_hook(module, input, output):
    if output.shape[0] < 99:
        forward_maps.append(output)
    return None


def display(forward_maps):
    for i, activations in enumerate(forward_maps):
        activationMap = torch.squeeze(activations[0])
        activationMap = activationMap.sum(0)
        activationMap = activationMap.data.cpu().numpy()
        # activationMap = cv2.resize(activationMap, (160, 80))
        if np.max(activationMap) - np.min(activationMap) != 0:
            activationMap = (activationMap - np.min(activationMap)) / (np.max(activationMap) - np.min(activationMap))
        if not activationMap.shape[0] == 80:
            activationMap_pad1 = np.zeros((80 - activationMap.shape[0], activationMap.shape[1]))
            activationMap_pad2 = np.zeros((80, 160 - activationMap.shape[1]))
            activationMap = np.concatenate((activationMap, activationMap_pad1), axis=-2)
            activationMap = np.concatenate((activationMap, activationMap_pad2), axis=-1)

        activationMaps = activationMap if i == 0 else np.concatenate([activationMaps, activationMap], axis=-2)
    cv2.imshow('', activationMaps)
    cv2.waitKey(1)


class Agent:
    def __init__(self, encoder, stack_num, state_dim, action_dim,
                 lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std):
        self.encoder = encoder
        self.buffer = RolloutBuffer()
        self.algorithm = PPO(stack_num, state_dim, action_dim,
                             lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                             action_std_init=action_std, buffer=self.buffer)

        for i, layers in enumerate(self.encoder._modules.items()):
            if i == 0:
                for layer in layers[1]:
                    if not isinstance(layer, GroupNorm):
                        layer.register_forward_hook(forward_hook)

    def sample_action(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.Tensor(obs)
            if len(obs.shape) < 4:
                obs = obs.unsqueeze(0)
        obs = obs.cuda()
        del forward_maps[:]
        with torch.no_grad():
            code = self.encoder(obs)
            out = self.algorithm.select_action(code)
        display(forward_maps)

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
