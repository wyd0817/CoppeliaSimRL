import time
import gym
from stable_baselines3 import DQN

import os
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import EvalCallback
from os.path import dirname, join, abspath
from os import path, pardir
from torch.utils.tensorboard import SummaryWriter
import sys
from datetime import datetime
import torch

utils_dir = join(abspath(join(dirname(__file__),pardir)),'utils')
sys.path.append(utils_dir)

tensorboard_log = join(abspath(join(dirname(__file__),pardir)),'logs_train')

# ---------------- create environment
env = gym.make('CartPole-v1')

# ---------------- callback functions
log_dir = join(abspath(join(dirname(__file__),pardir)),'CartPole/saved_models/tmp/DQN')
os.makedirs(log_dir, exist_ok=True)
env = Monitor(env, log_dir)

callback_save_best_model = EvalCallback(env, best_model_save_path=log_dir, log_path=log_dir, eval_freq=500, deterministic=True, render=False)
callback_list = CallbackList([callback_save_best_model])

TRAINING_MODE = False

# ---------------- model learning
if TRAINING_MODE == True:
    print('Learning the model')
    start_time =datetime.now()
    model = DQN(policy='MlpPolicy',
            env=env, 
            learning_rate=7e-4,
            policy_kwargs=dict(activation_fn=torch.nn.ReLU, net_arch=[256, 256]),
            verbose=2, 
            tensorboard_log = tensorboard_log)
    model.learn(total_timesteps=1E+5, callback=callback_list) 
    end_time =datetime.now()
    print('The training time: ',(end_time - start_time))
    print('Learning finished')

    del model
else:
    # ---------------- prediction
    print('Prediction')
    model_dir = join(abspath(join(dirname(__file__),pardir)),'CartPole/saved_models/tmp/DQN/best_model')
    model = DQN.load(model_dir, env=env)

    observation = env.reset()
    for i in range(1000):
        action, _state = model.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(action)
        env.render()
        if done:
            observation = env.reset()
        time.sleep(0.01)

    env.close()
