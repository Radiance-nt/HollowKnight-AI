"""https://github.com/nikhilbarhate99/PPO-PyTorch/"""
import os

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

################################## set device ##################################

print("============================================================================================")

# set device to cpu or cuda
device = torch.device('cpu')

if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

print("============================================================================================")


################################## PPO Policy ##################################


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def __len__(self):
        return len(self.states)

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class Actor(nn.Module):
    def __init__(self, state_dim, action_std_init):
        super(Actor, self).__init__()
        self.mlps = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh()
        )
        self.moves = nn.Sequential(
            nn.Linear(32, 8), nn.Tanh(),
            nn.Linear(8, 3), nn.Tanh(),
            nn.Softmax(dim=-1)
        )

        self.linear1 = nn.Sequential(
            nn.Linear(32, 12), nn.Tanh())

        self.linear2 = nn.Sequential(
            nn.Linear(15, 5), nn.Tanh())

    def forward(self, x):
        dense = self.mlps(x)
        moves = self.moves(dense)
        dense1 = self.linear1(dense)
        keys = self.linear2(torch.cat((dense1, moves), dim=1))

        return moves, keys


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init):
        super(ActorCritic, self).__init__()

        # if has_continuous_action_space:
        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # actor

        self.actor = Actor(state_dim, action_std_init)

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        moves, keys = self.actor(state)

        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        keys_dist = MultivariateNormal(keys, cov_mat)
        key_action = keys_dist.sample()
        key_action_logprob = keys_dist.log_prob(key_action)

        moves_dist = Categorical(moves)
        move_action = moves_dist.sample()
        move_action_logprob = moves_dist.log_prob(move_action)

        action = torch.cat((move_action, key_action.squeeze()))
        action_logprob = torch.cat((move_action_logprob, key_action_logprob))
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        moves, keys = self.actor(state)

        action_var = self.action_var.expand_as(keys)
        cov_mat = torch.diag_embed(action_var).to(device)
        keys_dist = MultivariateNormal(keys, cov_mat)

        # For Single Action Environments.
        if self.action_dim == 1:
            action = action.reshape(-1, self.action_dim)

        moves_dist = Categorical(moves)

        move_action = moves_dist.sample()
        move_action_logprob = moves_dist.log_prob(move_action)

        key_action = keys_dist.sample()
        key_action_logprob = keys_dist.log_prob(key_action)

        # action = torch.cat(move_action, key_action)
        action_logprob = torch.cat((move_action_logprob, key_action_logprob))

        dist_entropy = moves_dist.entropy() + keys_dist.entropy()

        state_values = self.critic(state)

        return action_logprob, state_values, dist_entropy


class PPO:
    def __init__(self, stack_num, state_dim, action_dim,
                 lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 action_std_init=0.6, buffer=None):

        self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = buffer
        self.capsule = nn.Sequential(nn.Flatten(start_dim=-2), nn.Linear(stack_num * state_dim, state_dim)).to(device)
        self.policy = ActorCritic(state_dim, action_dim, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if (self.action_std <= min_action_std):
            self.action_std = min_action_std
            print("setting actor output action_std to min_action_std : ", self.action_std)
        else:
            print("setting actor output action_std to : ", self.action_std)
        self.set_action_std(self.action_std)

    def select_action(self, state):

        with torch.no_grad():
            # state = torch.FloatTensor(state).to(device)
            cap_state = self.capsule(state).unsqueeze(0)
            action, action_logprob = self.policy_old.act(cap_state)

        self.buffer.states.append(state.cpu())
        self.buffer.actions.append(action.cpu())
        self.buffer.logprobs.append(action_logprob.cpu())

        return action.detach().cpu().numpy().flatten()
        # return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        length = min(len(self.buffer.rewards), len(self.buffer.is_terminals), len(self.buffer.states),
                     len(self.buffer.actions), len(self.buffer.logprobs))
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards[:length]),
                                       reversed(self.buffer.is_terminals[:length])):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        # convert list to tensor
        old_states = torch.squeeze(self.capsule(torch.stack(self.buffer.states[:length], dim=0).to(device))).detach()
        old_actions = torch.squeeze(torch.stack(self.buffer.actions[:length], dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs[:length], dim=0)).detach().to(device)
        loss = None
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs.resize_as(old_logprobs) - old_logprobs.detach()).mean(1)  # ToDo mean!

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
        infos = {}
        if loss is not None:
            infos['loss'] = loss.mean().item()
        return infos

    def save(self, path, episode):
        path = os.path.join(path, 'PPO_' + str(episode) + '.pkl')
        torch.save(self.policy_old.state_dict(), path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
