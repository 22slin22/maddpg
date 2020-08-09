import numpy as np
from PIL import Image, ImageDraw


class World:
    def __init__(self, players, landmarks, size):
        self.players = players
        self.landmarks = landmarks
        self.size = size

    def make_world(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def step(self):
        for agent in self.players:
            agent.pos += agent.vel
            # np.clip(agent.pos, agent.size, self.size - agent.size)

    def render(self):
        img = Image.new('RGB', (int(self.size), int(self.size)), color=(255, 255, 255))
        d = ImageDraw.Draw(img)
        for landmark in self.landmarks:
            d.ellipse(np.concatenate([landmark.pos - (landmark.size, landmark.size), landmark.pos + (landmark.size, landmark.size)]).tolist(), fill=landmark.color)
        for player in self.players:
            d.ellipse(np.concatenate([player.pos - (player.size, player.size), player.pos + (player.size, player.size)]).tolist(), fill=player.color)
        return img
