class Entity:
    def __init__(self, size, color):
        self.pos = None
        self.size = size
        self.color = color


class MovableEntity(Entity):
    def __init__(self, size, color):
        super(MovableEntity, self).__init__(size, color)
        self.vel = None


class Player(MovableEntity):
    def __init__(self, size=20, color='blue'):
        super(Player, self).__init__(size=size, color=color)


class Landmark(Entity):
    def __init__(self, size=20, color='green'):
        super(Landmark, self).__init__(size=size, color=color)
