import numpy as np
from environment.entities import Landmark, Player
from environment.world import World


class GetToLandmark:
    def __init__(self, n_agents, max_steps=60, world_size=600.0, vel_scale=10.0):
        self.n_agents = n_agents
        self.max_steps = max_steps
        self.vel_scale = vel_scale

        # Observation size = relative_landmark_position(2)
        self.obs_size = 2
        self.act_size = 2

        self.players = []
        for _ in range(n_agents):
            self.players.append(Player())
        self.landmark = Landmark()

        self.world = World(self.players, [self.landmark], size=world_size)

        self.step_counter = 0

    def reset(self):
        self.landmark.pos = np.random.rand(2) * self.world.size
        for player in self.players:
            player.pos = np.random.rand(2) * self.world.size
            player.vel = np.zeros(2, dtype="float32")
        self.step_counter = 0

        return self._get_obs()

    def step(self, actions):
        # print(actions)
        for act, player in zip(actions, self.players):
            player.vel = act * self.vel_scale

        self.world.step()

        rewards = self._calculate_rewards()

        self.step_counter += 1
        if self.step_counter >= self.max_steps:
            return self.reset(), rewards, True
        return self._get_obs(), rewards, False

    def _get_obs(self):
        obs = np.ndarray((self.n_agents, 2))
        for i, player in enumerate(self.players):
            obs[i] = self.landmark.pos - player.pos
        # print(obs)
        return obs

    def _calculate_rewards(self):
        r = np.ndarray((self.n_agents,))
        for i, player in enumerate(self.players):
            r[i] = -np.sqrt(np.sum(np.square(player.pos - self.landmark.pos))) * 0.001
        # print(r)
        return r

    def render(self):
        return self.world.render()
