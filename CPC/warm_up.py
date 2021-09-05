import math
import torch
import torchvision.transforms as transforms
from torch import nn
from CPC.Encoder import ResEncoder
from torchvision import datasets
import numpy as np

from CPC.tools import FrameDataset, GaussianBlur, TwoCropsTransform, SimSiam, Buffer

momentum = 0.9
weight_decay = 1e-4
batch_size = 256
lr = 0.05
init_lr = lr * batch_size / 256


# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
def to_tensor(pic):
    default_float_dtype = torch.get_default_dtype()
    img = torch.from_numpy(pic).contiguous()
    # backward compatibility
    if isinstance(img, torch.ByteTensor):
        return img.to(dtype=default_float_dtype).div(255)
    else:
        return img


augmentation = [
    to_tensor,
    transforms.RandomApply([transforms.GaussianBlur(3, [.1, 2.])], p=0.5),
    transforms.RandomResizedCrop((2, 160), scale=(0.2, 1.)),
    transforms.Normalize(mean=[0.445], std=[0.227])
]


# augmentation = [    transforms.RandomApply([
#         transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
#     ], p=0.8),
#         transforms.RandomGrayscale(p=0.2),z]


def adjust_learning_rate(optimizer, init_lr, epoch, epochs=100):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


def train(train_loader, model, criterion, optimizer):
    # switch to train mode
    model.train()

    for i, (images) in enumerate(train_loader):
        images[0] = images[0].to(torch.float32).cuda()
        images[1] = images[1].to(torch.float32).cuda()

        # compute output and loss
        p1, p2, z1, z2 = model(x1=images[0], x2=images[1])
        loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


def warm_up_cpc(simsiam, img_buffer, epoch=0, warm_up_episode=5000, writer=None):
    train_dataset = FrameDataset(img_buffer, TwoCropsTransform(transforms.Compose(augmentation)))
    optimizer = torch.optim.SGD(simsiam.parameters(), init_lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    criterion = nn.CosineSimilarity(dim=1).cuda()
    total_episode = epoch * warm_up_episode
    losses = 0
    for episode in range(warm_up_episode):
        # adjust_learning_rate(optimizer, init_lr, epoch)
        loss = train(train_loader, simsiam, criterion, optimizer)
        if episode % 50 == 0:
            print('Warming up CPC: %s/%s episode in %s epoch.' % (episode, warm_up_episode, epoch))
        if writer:
            writer.add_scalar('CPC' + ' /' + 'loss', loss, total_episode)
        losses += loss
        total_episode += 1
    return losses / warm_up_episode


def warm_up_process(simsiam, img_buffer, warm_up_epoch, warm_up_episode, writer, cpc_model_name):
    epoch = 0
    best_loss = 999
    while epoch < warm_up_epoch:
        loss = warm_up_cpc(simsiam, img_buffer,
                           epoch=epoch, warm_up_episode=warm_up_episode, writer=writer)
        print('CPC Epoch %s: loss = %s' % (epoch, loss))
        if loss < best_loss:
            best_loss = loss
            torch.save(simsiam, cpc_model_name)
            print('Best CPC Loss update: ', loss)
        epoch += 1


if __name__ == "__main__":
    state_dim = 256
    pred_dim = 64
    encoder = ResEncoder(in_channels=1, out_dims=state_dim).cuda()
    simsiam = SimSiam(encoder, state_dim, pred_dim).cuda()
    img_buffer = Buffer(_length=1, _max_replay_buffer_size=3000)
    for i in range(50):
        img_buffer.append(np.ones((80, 160)))

    warm_up_cpc(simsiam, img_buffer)
