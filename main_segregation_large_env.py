import marl
from custom.envs.corridor_env import CorridorEnv
from custom.envs.segregation_env import SegregationEnv
from custom.envs.test_env import TestEnv
from marl.agent import DQNAgent
from marl.model.nn import FootCnn, MlpNet

import gym

AGENT_COUNT = 200

env = SegregationEnv(size=20, entity_count=AGENT_COUNT)

obs_s = env.observation_space
act_s = env.action_space

agents = [DQNAgent(MlpNet(9, 4,[128,64]), obs_s, act_s, name=f"Segregation_{i}") for i in range(AGENT_COUNT)]

for i,a in enumerate(agents):
   a.policy.load(filename=f"models/marl-Segregation_{i%12}-200000")

mas = marl.MARL(agents)

#mas.learn(env, nb_timesteps=200000, save_freq=50000, test_freq=1000)

# Test the agent for 1000 episodes
mas.test(env, nb_episodes=10000)