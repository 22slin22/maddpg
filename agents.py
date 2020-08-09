import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.autograd import Variable
from torch import Tensor
from networks import ActorNet, CriticNet
from utils import utils


class DDPGAgent:
    def __init__(self, obs_size, act_size, critic_input_size, hidden_layer_size, lr, gamma, tau, exploration_decay):
        super(DDPGAgent, self).__init__()
        self.gamma = gamma
        self.tau = tau
        self.exploration_decay = exploration_decay

        self.actor = ActorNet(obs_size, act_size, hidden_layer_size)
        self.actor_target = ActorNet(obs_size, act_size, hidden_layer_size)
        utils.hard_update(self.actor_target, self.actor)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr)

        self.exploration_noise = utils.OUNoise(act_size)

        self.critic = CriticNet(critic_input_size, hidden_layer_size)
        self.critic_target = CriticNet(critic_input_size, hidden_layer_size)
        utils.hard_update(self.critic_target, self.critic)

        self.critic_criterion = nn.MSELoss()
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)

    def act(self, obs, explore=True):
        action = self.actor.forward(Tensor(obs))
        if explore:
            action += Variable(Tensor(self.exploration_noise.get_next_noise()))
            action = action.clamp(-1.0, 1.0)
            self.exploration_noise.scale *= self.exploration_decay
        return action

    def _soft_update_target_nets(self):
        utils.soft_update(self.actor_target, self.actor, self.tau)
        utils.soft_update(self.critic_target, self.critic, self.tau)

    def train(self, obs, act, rew, next_obs, done):
        obs = torch.from_numpy(obs)
        act = torch.from_numpy(act)
        rew = torch.from_numpy(rew).unsqueeze(1)
        next_obs = torch.from_numpy(next_obs)
        done = torch.from_numpy(done.astype(np.float)).unsqueeze(1)
        # act = torch.unsqueeze(act, 1)
        qs = self.critic.forward(torch.cat((obs, act), axis=1))
        # print(qs)
        next_actions = self.actor_target.forward(next_obs)
        next_qs = self.critic_target.forward(torch.cat((next_obs, next_actions.detach()), dim=1))

        # print("Next qs", next_qs)
        # print("rew", rew)
        targets = rew + self.gamma * next_qs * (1 - done)
        # print(targets)

        critic_loss = self.critic_criterion(qs, targets)
        # print(critic_loss)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -torch.mean(self.critic.forward(torch.cat([obs, self.actor.forward(obs)], 1)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update_target_nets()
