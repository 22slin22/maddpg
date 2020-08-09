from collections import deque
import random
import numpy as np


class ReplayBuffer:
    def __init__(self, max_len):
        self.obs = deque(maxlen=max_len)
        self.acts = deque(maxlen=max_len)
        self.rews = deque(maxlen=max_len)
        self.next_obs = deque(maxlen=max_len)
        self.dones = deque(maxlen=max_len)

    def add_experience(self, obs, acts, rews, next_obs, done):
        self.obs.append(obs)
        self.acts.append(acts)
        self.rews.append(rews)
        self.next_obs.append(next_obs)
        self.dones.append(done)

    def sample(self, batch_size):
        idx = random.sample(range(len(self)), batch_size)
        obs = np.array([self.obs[i] for i in idx], dtype="float32")
        acts = np.array([self.acts[i] for i in idx], dtype="float32")
        rews = np.array([self.rews[i] for i in idx], dtype="float32")
        next_obs = np.array([self.next_obs[i] for i in idx], dtype="float32")
        dones = np.array([self.dones[i] for i in idx])
        return obs, acts, rews, next_obs, dones

    def __len__(self):
        return len(self.obs)
