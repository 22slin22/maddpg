import torch
from torch import nn
import torch.nn.functional as F


class CriticNet(nn.Module):
    def __init__(self, critic_input_size, hidden_layer_size):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(critic_input_size, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ActorNet(nn.Module):
    def __init__(self, obs_size, act_size, hidden_layer_size):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(obs_size, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, act_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x
