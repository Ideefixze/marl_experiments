import random

import gym
import numpy as np
from PIL import Image
import itertools
import cv2
import math

from custom.core import Creature, Grid


class TestEnv(gym.Env):

    def __init__(self, size=8, entity_count=2):
        self.size = size
        self.entity_count = entity_count
        self._max_step_count = 15
        self.reset()

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0,high=self.size,shape=(2*entity_count,))


    def reset(self):
        self.grid = Grid(self.size)
        self.entities = [Creature(color=((255,0,0) if i == 0 else (0,255,0))) for i in range(self.entity_count)]
        for e in self.entities:
            self.grid.add_new_creature(e) #register each creature to the grid
        self.grid.init()
        self.entities[0].x = 0
        self.entities[0].y = self.size-1
        self.entities[1].x = self.size-1
        self.entities[1].y = 0
        self._current_step_count = 0

        return [self.state()]*self.entity_count

    def state(self):
        positions = []
        for entity in self.entities:
            positions.append(entity.x)
            positions.append(entity.y)

        positions = np.array(positions)
        return positions

    def step(self, action_n):

        #Perform actions
        for i, action in enumerate(action_n):
            x = self.entities[i].x
            y = self.entities[i].y

            self.entities[i].action(action)


        obs_n = [self.state()]*self.entity_count

        done = [False if self._current_step_count < self._max_step_count else True] * self.entity_count

        reward_n = list()
        for i, entity in enumerate(self.entities):
            if entity.x == 0 and entity.y == 0:
                reward_n.append(random.random())
            else:
                reward_n.append(0)

        self._current_step_count += 1

        return obs_n, np.array(reward_n), done, {}

    def render(self):
        img = self.grid.get_image()
        img = img.resize((400, 400), Image.NEAREST)
        cv2.imshow("Environment", np.array(img))
        cv2.waitKey(100)

    """
    def get_image(self):
        env = np.zeros((self.size, self.size, 3), dtype=np.uint8)

        border_color = (128, 128, 128)
        env[:, 0] = border_color
        env[:, self.size-1] = border_color
        env[0, :] = border_color
        env[self.size - 1, :] = border_color

        for entity in self.entities:
            env[entity.x][entity.y] = entity.color

        img = Image.fromarray(env, 'RGB')
        return img
    """
