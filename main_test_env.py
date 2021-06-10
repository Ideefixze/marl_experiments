import marl
from marl.agent import DQNAgent
from marl.model.nn import MlpNet

from custom.envs.test_env import TestEnv

import gym

env = TestEnv(size=5, entity_count=2)

obs_s = env.observation_space
act_s = env.action_space

print(obs_s)
print(act_s)

ag1 = DQNAgent("MlpNet", obs_s, act_s, name="Bob")
ag2 = DQNAgent("MlpNet", obs_s, act_s, name="Jack")

mas = marl.MARL([ag1, ag2])

mas.learn(env, nb_timesteps=4000)

# Test the agent for 10 episodes
mas.test(env, nb_episodes=100)