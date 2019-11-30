import os
import gym
import numpy as np
import math

from stable_baselines import results_plotter
from stable_baselines import A2C, SAC, PPO2, TD3
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.sac.policies import MlpPolicy as SacMlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy

import matplotlib.pyplot as plt
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)


best_mean_reward, n_steps = -np.inf, 0

def callback(_locals, _globals):
  global n_steps, best_mean_reward
  # Print stats every 10000 calls
  if (n_steps + 1) % 10000 == 0:
      # Evaluate policy training performance
      x, y = ts2xy(load_results(log_dir), 'timesteps')
      if len(x) > 0:
          mean_reward = np.mean(y[-100:])
          print(x[-1], 'timesteps')
          print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

          # New best model, you could save the agent here
          if mean_reward > best_mean_reward:
              best_mean_reward = mean_reward
              # Example for saving best model
              print("Saving new best model")
              _locals['self'].save(log_dir + 'best_model.pkl')
  n_steps += 1
  return True


if __name__ == '__main__':

    env_names = ['MountainCarContinuous-v0']#, 'BipedalWalker-v2', 'LunarLanderContinuous-v2']
    n_runs = 4
    train_timestamps = int(1e5)

    for env_name in env_names:
        for i in range(n_runs):
            print('Run number: ', i)
            
            agent_name = 'a2c'
            print('Testing agent ' + agent_name)
            best_mean_reward, n_steps = -np.inf, 0
            log_dir = 'log_{}_{}_{}/'.format(agent_name, env_name, i)
            os.makedirs(log_dir, exist_ok=True)
            env = gym.make(env_name)
            env = Monitor(env, log_dir, allow_early_resets=True)
            env = DummyVecEnv([lambda: env])
            model_A2C = A2C(MlpPolicy, env, verbose=0)
            model_A2C.learn(total_timesteps=train_timestamps, callback=callback)
        
            agent_name = 'sac'
            print('Testing agent ' + agent_name)
            best_mean_reward, n_steps = -np.inf, 0
            log_dir = 'log_{}_{}_{}/'.format(agent_name, env_name, i)
            os.makedirs(log_dir, exist_ok=True)
            env = gym.make(env_name)
            env = Monitor(env, log_dir, allow_early_resets=True)
            env = DummyVecEnv([lambda: env])
            model_SAC = SAC(SacMlpPolicy, env, verbose=0)
            model_SAC.learn(total_timesteps=train_timestamps, callback=callback)

            agent_name = 'ppo'
            print('Testing agent ' + agent_name)
            best_mean_reward, n_steps = -np.inf, 0
            log_dir = 'log_{}_{}_{}/'.format(agent_name, env_name, i)
            os.makedirs(log_dir, exist_ok=True)
            env = gym.make(env_name)
            env = Monitor(env, log_dir, allow_early_resets=True)
            env = DummyVecEnv([lambda: env])
            model_PPO2 = PPO2(MlpPolicy, env, verbose=0)
            model_PPO2.learn(total_timesteps=train_timestamps, callback=callback)
        


