import random

import gym
import numpy as np
from PIL import Image
import itertools
import cv2
import math

from custom.core import Creature, Grid


class SegregationEnv(gym.Env):

    def __init__(self, size=8, entity_count=2):
        self.size = size
        self.entity_count = entity_count
        self._max_step_count = size*5
        self.reset()

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(9,))


    def reset(self):
        self.grid = Grid(self.size)
        self.entities = [Creature(color=((255,0,0) if i % 2 == 0 else (0,0,255))) for i in range(self.entity_count)]
        for e in self.entities:
            self.grid.add_new_creature(e) #register each creature to the grid
        self.grid.init()

        self._current_step_count = 0

        return self.state()

    def state(self):

        grid = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        for e in self.entities:
            grid[e.x][e.y] = e.color

        all_data = []
        for e in self.entities:
            data = []

            for x,y in [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1), (0,0)]:
                try:
                    data.append(1 if (tuple(grid[e.x+x][e.y+y]) == e.color) else (0 if tuple(grid[e.x+x][e.y+y]) == (0,0,0) else -1))
                except:
                    data.append(0)
            all_data.append(data)

        #print(np.array(all_data))
        return np.array(all_data)

    def step(self, action_n):

        #Perform actions
        for i, action in enumerate(action_n):
            x = self.entities[i].x
            y = self.entities[i].y

            self.entities[i].action(action)


        obs_n = self.state()

        done = [False if self._current_step_count < self._max_step_count else True] * self.entity_count

        reward_n = list()
        seg_total = 0
        for i, entity in enumerate(self.entities):
            num_same_neighbours = len(list(filter(lambda x: x==1, obs_n[i])))
            num_opposite_neighbours = len(list(filter(lambda x: x == -1, obs_n[i])))
            seg_total += (num_same_neighbours-num_opposite_neighbours-1) / self.entity_count / 4
            reward_n.append((num_same_neighbours-num_opposite_neighbours-1))

        for i, entity in enumerate(self.entities):
            reward_n[i] = reward_n[i] + seg_total

        self._current_step_count += 1

        return obs_n, np.array(reward_n), done, {}

    def render(self):
        img = self.grid.get_image()
        img = img.resize((400, 400), Image.NEAREST)
        cv2.imshow("Environment", np.array(img))
        cv2.waitKey(100)
