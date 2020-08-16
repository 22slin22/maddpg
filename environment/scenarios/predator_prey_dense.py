import numpy as np
from environment.entities import Player
from environment.world import World


class PredatorPreyDense:
    def __init__(self, n_predators, n_preys, max_steps=60, world_size=600.0, vel_scale_predators=10.0, vel_scale_preys=15.0):
        self.n_predators = n_predators
        self.n_preys = n_preys
        self.max_steps = max_steps

        self.obs_size = self.n_players * 2
        self.act_size = 2

        self.predators = []
        for _ in range(n_predators):
            self.predators.append(Player(max_speed=vel_scale_predators, color="red"))
        self.preys = []
        for _ in range(n_preys):
            self.preys.append(Player(max_speed=vel_scale_preys, color="blue"))

        self.world = World(self.players, landmarks=[], size=world_size)

        self.step_counter = 0

    @property
    def players(self):
        return self.predators + self.preys

    @property
    def n_players(self):
        return self.n_predators + self.n_preys

    def reset(self):
        for player in self.players:
            player.pos = np.random.rand(2) * self.world.size
            player.vel = np.zeros(2, dtype="float32")
        self.step_counter = 0

        return self._get_obs()

    def step(self, actions):
        for act, player in zip(actions, self.players):
            vel = act * player.max_speed
            if np.sqrt(np.sum(np.square(vel))) > player.max_speed:
                vel = vel / (np.sqrt(np.sum(np.square(vel))) * player.max_speed)
            player.vel = vel

        self.world.step()

        rewards = self._calculate_rewards()

        self.step_counter += 1
        if self.step_counter >= self.max_steps:
            return self.reset(), rewards, True
        return self._get_obs(), rewards, False

    def _get_obs(self):
        obs = np.ndarray((self.n_players, 2 * self.n_players))
        for i, player in enumerate(self.players):
            obs_i = []
            for j, other in enumerate(self.players):
                if i == j:
                    obs_i.extend(self.world.get_normalized_pos(player))
                else:
                    obs_i.extend(self.world.get_normalized_pos(other) - self.world.get_normalized_pos(player))
            obs[i] = obs_i
        # print(obs)
        return obs

    def _calculate_rewards(self):
        r = np.zeros((self.n_players,), dtype="float32")
        # Calculate predator reward
        # Rewards base on nearest prey to every predator
        # Extra reward for touching
        predator_reward = 0
        for predator in self.predators:
            predator_reward -= min([predator.get_distance_to(prey) for prey in self.preys]) * 0.001
            # for prey in self.preys:
            #     if predator.is_colliding(prey):
            #         predator_reward += 10
        r[:self.n_predators] = predator_reward

        # Calculate prey reward
        # Reward based on sum of distances to every predator
        for i, prey in enumerate(self.preys):
            # See if predator is touching prey
            r[self.n_predators + i] = sum([prey.get_distance_to(predator) for predator in self.predators]) * 0.001

        return r

    def render(self):
        return self.world.render()
