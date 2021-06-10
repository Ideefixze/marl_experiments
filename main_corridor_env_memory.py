import torch
from torch import nn

import marl
from custom.envs.corridor_env import CorridorEnv
from custom.envs.test_env import TestEnv
from custom.modules import ModuleLSTM
from marl.agent import DQNAgent
from marl.model.nn import FootCnn, MlpNet

import gym



AGENT_COUNT = 4

env = CorridorEnv(size=8, entity_count=AGENT_COUNT,top_block=2, bot_block=5)

obs_s = env.observation_space
act_s = env.action_space

agents = [DQNAgent(ModuleLSTM(24, 7), obs_s, act_s, name=f"Corridor_const_pos_mems_total_{i}") for i in range(AGENT_COUNT)]

#for a in agents:
   #a.policy.load(filename="models/marl-Guy0-150000")

mas = marl.MARL(agents)

mas.learn(env, nb_timesteps=100000, save_freq=10000, test_freq=1000)

# Test the agent for 1000 episodes
mas.test(env, nb_episodes=10000)