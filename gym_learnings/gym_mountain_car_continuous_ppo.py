import time
import gym
from stable_baselines3 import PPO

import os
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import EvalCallback
from os.path import dirname, join, abspath
from os import path, pardir
from stable_baselines3.common.env_util import make_vec_env
from torch.utils.tensorboard import SummaryWriter
import sys
from datetime import datetime
import torch

utils_dir = join(abspath(join(dirname(__file__),pardir)),'utils')
sys.path.append(utils_dir)

tensorboard_log = join(abspath(join(dirname(__file__),pardir)),'logs_train/MountainCarContinuous')
log_dir = join(abspath(join(dirname(__file__),pardir)),'models/MountainCarContinuous/PPO')
os.makedirs(log_dir, exist_ok=True)

TRAINING_MODE = False
TRAINING_MODE = True

# ---------------- model learning
if TRAINING_MODE == True:
    # ---------------- create environment
    env = make_vec_env('MountainCarContinuous-v0',
                    n_envs = 4,
                    seed = 1,
                    monitor_dir = log_dir)
    print('Learning the model')
    start_time =datetime.now()
    model = PPO(policy='MlpPolicy',
            env=env, 
            learning_rate=7.77e-05,
            policy_kwargs=dict(log_std_init=-3.29, ortho_init=False),
            normalize_advantage = True,
            batch_size = 256,
            n_steps = 8,
            gamma = 0.9999,
            ent_coef = 0.00429,
            clip_range = 0.1,
            n_epochs = 10,
            gae_lambda = 0.9,
            max_grad_norm = 5,
            vf_coef = 0.19,
            use_sde = True,
            verbose=2, 
            # seed = 1,
            tensorboard_log = tensorboard_log)
    callback_save_best_model = EvalCallback(env, best_model_save_path=log_dir, log_path=log_dir, eval_freq=500, deterministic=True, render=False)
    callback_list = CallbackList([callback_save_best_model])
    model.learn(total_timesteps=20000.0, callback=callback_list) 
    end_time =datetime.now()
    print('The training time: ',(end_time - start_time))
    print('Learning finished')

    del model
else:
    # ---------------- create environment
    env = gym.make('MountainCarContinuous-v0')
    # ---------------- prediction
    print('Prediction')
    model_dir = join(abspath(join(dirname(__file__),pardir)),'models/MountainCarContinuous/PPO/best_model')
    model = PPO.load(model_dir, env=env)
    # print(model_dir)
    # print(env.observation_space)
    # print(env.observation_space.shape)
    # print(env.observation_space.shape[0])
    # print(env.action_space)

    observation = env.reset()
    for i in range(1000):
        action, _state = model.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(action)
        env.render()
        if done:
            observation = env.reset()
        time.sleep(0.01)

    env.close()
