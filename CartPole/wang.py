import numpy as np
import gym
import time
from gym.envs.mujoco.mujoco_env import MujocoEnv

env = gym.make('Swimmer-v3')

env = observation = env.reset()
for i in range(1000):
    observation, reward, done, info = env.step([1,2,3,4,5,6,7])
    env.render()
    if done:
        observation = env.reset()

    time.sleep(0.01)

env.close()
