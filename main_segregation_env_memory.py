import marl
from custom.envs.corridor_env import CorridorEnv
from custom.envs.segregation_env import SegregationEnv
from custom.envs.test_env import TestEnv
from custom.modules import ModuleLSTM
from marl.agent import DQNAgent
from marl.model.nn import FootCnn, MlpNet

import gym

AGENT_COUNT = 8

env = SegregationEnv(size=10, entity_count=AGENT_COUNT)

obs_s = env.observation_space
act_s = env.action_space

agents = [DQNAgent(ModuleLSTM(9, 4, 32), obs_s, act_s, name=f"Segregation_mem_large_{i}") for i in range(AGENT_COUNT)]

#for a in agents:
   #a.policy.load(filename="models/marl-Segregation_mem_0-100000")

mas = marl.MARL(agents)

mas.learn(env, nb_timesteps=10000, save_freq=10000, test_freq=1000)

# Test the agent for 1000 episodes
mas.test(env, nb_episodes=10000)