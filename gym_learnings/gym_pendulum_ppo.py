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

tensorboard_log = join(abspath(join(dirname(__file__),pardir)),'logs_train/Pendulum')
log_dir = join(abspath(join(dirname(__file__),pardir)),'models/Pendulum/PPO')
os.makedirs(log_dir, exist_ok=True)

TRAINING_MODE = False
# TRAINING_MODE = True


# ---------------- model learning
if TRAINING_MODE == True:
    # ---------------- create environment
    env = make_vec_env('Pendulum-v1',
                    n_envs = 4,
                    seed = 1,
                    monitor_dir = log_dir)
    print('Learning the model')
    start_time =datetime.now()
    model = PPO(policy='MlpPolicy',
                env=env, 
                n_steps = 1024,
                gae_lambda = 0.95,
                gamma = 0.9,
                n_epochs = 10,
                ent_coef = 0.0,
                learning_rate = 1e-3,
                clip_range = 0.2,
                use_sde = True,
                sde_sample_freq = 4,
                verbose=2, 
                # seed = 1,
                tensorboard_log = tensorboard_log)


    callback_save_best_model = EvalCallback(env, best_model_save_path=log_dir, log_path=log_dir, eval_freq=500, deterministic=True, render=False)
    callback_list = CallbackList([callback_save_best_model])
    model.learn(total_timesteps=1e5, callback=callback_list) 
    end_time =datetime.now()
    print('The training time: ',(end_time - start_time))
    print('Learning finished')

    del model
else:
    # ---------------- create environment
    env = gym.make('Pendulum-v1')
    # ---------------- prediction
    print('Prediction')
    model_dir = join(abspath(join(dirname(__file__),pardir)),'models/Pendulum/PPO/best_model')
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
