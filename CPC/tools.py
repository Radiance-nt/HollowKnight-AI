import cv2
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import ConcatDataset, Dataset
from PIL import ImageFilter
import random


class Buffer:
    def __init__(self, _length, _stride=4, _max_replay_buffer_size=300):
        self.buffer = []
        self._max_replay_buffer_size = _max_replay_buffer_size
        self._top = 0
        self._length = _length
        self._stride = _stride
        self.init_buffer()

    def append(self, x):
        if self.__len__() < self._max_replay_buffer_size:
            self.buffer.append(x)
            self._top = (self._top + 1) % self._max_replay_buffer_size

        else:
            self.buffer[self._top] = x
            self._top = (self._top + 1) % self._max_replay_buffer_size

    def get_stack(self, top=None, length=None, stride=None):
        if top is None:
            top = self._top
        if length is None:
            length = self._length
        if stride is None:
            stride = self._stride
        stack = []
        for i in range(length):
            pointer = top - i * stride
            pointer = pointer + self._max_replay_buffer_size if pointer < 0 else pointer
            stack.append(self.buffer[pointer])
        return np.stack(stack)[:, np.newaxis, :, :]
        # return np.concatenate(stack)

    def init_buffer(self):
        self.buffer = [np.zeros((80, 160))] * self._max_replay_buffer_size

    def __getitem__(self, index):
        return self.get_stack(index)

    def __len__(self):
        return len(self.buffer)


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class FrameDataset(Dataset):
    def __init__(self, buffer, transforms=None, **kwargs):
        self.buffer = buffer
        self.transform = transforms

    def __getitem__(self, index: int):
        sample = self.buffer[index]
        # sample = torch.tensor(sample)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.buffer)


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """

    def __init__(self, base_encoder, dim=256, pred_dim=64):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True),  # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True),  # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False))  # output layer
        self.encoder.fc[6].bias.requires_grad = False  # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                       nn.BatchNorm1d(pred_dim),
                                       nn.ReLU(inplace=True),  # hidden layer
                                       nn.Linear(pred_dim, dim))  # output layer

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.encoder(x1)  # NxC
        z2 = self.encoder(x2)  # NxC

        p1 = self.predictor(z1)  # NxC
        p2 = self.predictor(z2)  # NxC

        return p1, p2, z1.detach(), z2.detach()
