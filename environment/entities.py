import numpy as np


class Entity:
    def __init__(self, size, color):
        self.pos = None
        self.size = size
        self.color = color

    def get_distance_to(self, other):
        if self.pos is None or other.pos is None:
            raise ValueError("The position of both entities has to be initialized")
        return np.sqrt(np.sum(np.square(self.pos - other.pos)))

    def is_colliding(self, other):
        return self.get_distance_to(other) < self.size + other.size


class MovableEntity(Entity):
    def __init__(self, size, color):
        super(MovableEntity, self).__init__(size, color)
        self.vel = None


class Player(MovableEntity):
    def __init__(self, size=20, color='blue', max_speed=None):
        super(Player, self).__init__(size=size, color=color)
        self.max_speed = max_speed


class Landmark(Entity):
    def __init__(self, size=20, color='green'):
        super(Landmark, self).__init__(size=size, color=color)
