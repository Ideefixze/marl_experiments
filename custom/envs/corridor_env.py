import random

import gym
import numpy as np
from PIL import Image
import itertools
import cv2
import math

from custom.corridor_core import CorridorGrid, CreatureCameleon


class CorridorEnv(gym.Env):

    def __init__(self, size=8, entity_count=2, top_block=3, bot_block=6):
        self.size = size
        self.entity_count = entity_count
        self._max_step_count = size*3
        self.top_block = top_block
        self.bot_block = bot_block
        self.reset()

        self.action_space = gym.spaces.Discrete(7)
        self.observation_space = gym.spaces.Box(low=np.array([0,0,0,0,0,0]*entity_count),high=np.array([self.size,self.size,1,1,1,1]*entity_count),shape=(6*entity_count,))

    def reset(self):
        self.grid = CorridorGrid(self.size, self.top_block, self.bot_block)
        self.entities = [CreatureCameleon() for i in range(self.entity_count)]
        for e in self.entities:
            self.grid.add_new_creature(e) #register each creature to the grid
        self.grid.init()
        self._current_step_count = 0

        self.starting_side = []
        for e in self.entities:
            if e.x < self.size/2:
                self.starting_side.append("left")
            else:
                self.starting_side.append("right")

        return [self.state()]*self.entity_count

    def state(self):
        data = []
        for i,entity in enumerate(self.entities):
            data.append(entity.x)
            data.append(entity.y)
            data.append(entity.color[0]//255)
            data.append(entity.color[1]//255)
            data.append(entity.color[2]//255)
            data.append(0 if self.starting_side[i]=="left" else 1)

        positions = np.array(data)
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

        """
        c = 0
        for i,side in enumerate(self.starting_side):
            if self.entities[i].x == self.size-1 and side=="left":
                reward_n.append(1)
                c += 1
            elif self.entities[i].x < self.size//2 and side == "left":
                reward_n.append(-0.1)
            elif self.entities[i].x == 0 and side=="right":
                reward_n.append(1)
                c += 1
            elif self.entities[i].x > self.size//2 and side=="right":
                reward_n.append(-0.1)
            else:
                reward_n.append(0)
        """

        c = 0
        for i,side in enumerate(self.starting_side):
            if self.entities[i].x == self.size-1 and side=="left":
                c += 1
            elif self.entities[i].x == 0 and side=="right":
                c += 1
            else:
                c -= 0.05

        for e in self.entities:
            reward_n.append(c)

        self._current_step_count += 1

        return obs_n, np.array(reward_n), done, {}

    def render(self):
        img = self.grid.get_image()
        img = img.resize((400, 400), Image.NEAREST)
        cv2.imshow("Environment", np.array(img))
        cv2.waitKey(100)

