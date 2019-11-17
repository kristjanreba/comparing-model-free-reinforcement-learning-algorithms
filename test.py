import os
import gym
import numpy as np

from stable_baselines import results_plotter
from stable_baselines import A2C, SAC, PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.sac.policies import MlpPolicy as SacMlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy

import matplotlib.pyplot as plt
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)


def render_env(env, model, timestamps):
    obs = env.reset()
    for i in range(timestamps):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
    env.close()


best_mean_reward, n_steps = -np.inf, 0

def callback(_locals, _globals):
  """
  Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
  :param _locals: (dict)
  :param _globals: (dict)
  """
  global n_steps, best_mean_reward
  # Print stats every 1000 calls
  if (n_steps + 1) % 1000 == 0:
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


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, agent_name, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(agent_name)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.savefig(agent_name+".png")


def plot_all_together(agent_names, env_name, n_runs, title):
    fig = plt.figure(title)
    for a in agent_names:
        x, y = ts2xy(load_results('log_{}_{}_{}/'.format(agent_name, env_name, run_number)), 'timesteps')
        mean = []
        std = []
        #y = moving_average(y, window=50)
        #x = x[len(x) - len(y):]
        plt.plot(x, y, label=a.upper())
        #plt.fill_between(x, mean-std, mean+std)


    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title)
    plt.legend()
    plt.savefig(title + ".png")



if __name__ == '__main__':

    retrain_agents = True
    RENDER_TIMESTAMPS = int(1e3)
    TRAIN_TIMESTAMPS = int(1e5)
    n_runs = 10
    env_name = 'MountainCarContinuous-v0'
    #env_name = 'BipedalWalker-v2'
    #env_name = 'Pong-v0'

    if retrain_agents:
        for i in range(n_runs):
            agent_name = 'a2c'
            print('Testing agent ' + agent_name)
            best_mean_reward, n_steps = -np.inf, 0
            log_dir = 'log_{}_{}_{}/'.format(agent_name, env_name, run_number)
            os.makedirs(log_dir, exist_ok=True)
            env = gym.make(env_name)
            env = Monitor(env, log_dir, allow_early_resets=True)
            env = DummyVecEnv([lambda: env])
            model_A2C = A2C(MlpPolicy, env, verbose=0)
            model_A2C.learn(total_timesteps=TRAIN_TIMESTAMPS, callback=callback)
            plot_results(log_dir, agent_name)
        
            agent_name = 'sac'
            print('Testing agent ' + agent_name)
            best_mean_reward, n_steps = -np.inf, 0
            log_dir = 'log_{}_{}_{}/'.format(agent_name, env_name, run_number)
            os.makedirs(log_dir, exist_ok=True)
            env = gym.make(env_name)
            env = Monitor(env, log_dir, allow_early_resets=True)
            env = DummyVecEnv([lambda: env])
            model_SAC = SAC(SacMlpPolicy, env, verbose=0)
            model_SAC.learn(total_timesteps=TRAIN_TIMESTAMPS, callback=callback)
            plot_results(log_dir, agent_name)

            agent_name = 'ppo'
            print('Testing agent ' + agent_name)
            best_mean_reward, n_steps = -np.inf, 0
            log_dir = 'log_{}_{}_{}/'.format(agent_name, env_name, run_number)
            os.makedirs(log_dir, exist_ok=True)
            env = gym.make(env_name)
            env = Monitor(env, log_dir, allow_early_resets=True)
            env = DummyVecEnv([lambda: env])
            model_PPO2 = PPO2(MlpPolicy, env, verbose=0)
            model_PPO2.learn(total_timesteps=TRAIN_TIMESTAMPS, callback=callback)
            plot_results(log_dir, agent_name)



    ##########################################################################
    # EXPERIMENTS Timestaps vs Reward vs Robustness to initialization
    ##########################################################################
    agent_names = ['a2c','ppo','sac']
    title = env_name
    def plot_all_together(agent_names, env_name, n_runs, title)


