import numpy as np


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class OUNoise:
    def __init__(self, act_size, scale=0.4, mu=0, theta=0.15, sigma=0.2):
        self.act_size = act_size
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.act_size) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.act_size) * self.mu

    def get_next_noise(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.act_size)
        self.state = self.state + dx
        noise = self.state * self.scale
        return noise
