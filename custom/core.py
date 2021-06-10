import random

import numpy as np
from PIL import Image


class Grid:
    def __init__(self, size):
        self.size = size
        self.creatures = []
        self.blocked = set()

    def free_at(self, pos:(int,int)):
        x = pos[0]
        y = pos[1]

        if x>=0 and x<self.size and y>=0 and y<self.size:
            if pos not in self.blocked:
                if pos not in {(c.x, c.y) for c in self.creatures}:
                    return True

        return False

    def add_new_creature(self, creature:'Creature'):
        creature.grid = self
        self.creatures.append(creature)

    def add_new_blocked(self, blocked:(int,int)):
        self.blocked.add(blocked)

    def randomize_position(self, creature):
        creature.x = random.randint(0, self.size-1)
        creature.y = random.randint(0, self.size-1)

    def init(self):
        for creature in self.creatures:
            self.randomize_position(creature)
            while creature.position in self.blocked or creature.position in {(c.x, c.y) for c in self.creatures if c != creature}:
                self.randomize_position(creature)


    def get_image(self):
        env = np.zeros((self.size, self.size, 3), dtype=np.uint8)

        for blocked in self.blocked:
            env[blocked[0]][blocked[1]] = (128,128,128)

        for entity in self.creatures:
            env[entity.x][entity.y] = entity.color

        img = Image.fromarray(env, 'RGB')
        return img


class Creature:

    MOVE_LIST={0: (1,0), 1: (-1,0), 2: (0,1), 3: (0,-1)}

    def __init__(self, color=(255,0,0)):
        self.x = 0
        self.y = 0
        self.color = color
        self.grid = None

    @property
    def position(self):
        return self.x, self.y

    def __str__(self):
        return f"Creature ({self.x}, {self.y})"

    def action(self, choice):
        '''
        Gives us 4 total movement options. (0,1,2,3)
        '''
        if choice == 0:
            self.move(x=1, y=0)
        elif choice == 1:
            self.move(x=-1, y=0)
        elif choice == 2:
            self.move(x=0, y=1)
        elif choice == 3:
            self.move(x=0, y=-1)
        else:
            return False
        return True

    def move(self, x=False, y=False):
        if self.grid.free_at((self.x+x, self.y+y)):
            self.x += x
            self.y += y

