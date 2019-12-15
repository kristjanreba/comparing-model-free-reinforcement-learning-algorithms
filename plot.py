import os
import numpy as np
import math
from stable_baselines.results_plotter import load_results, ts2xy
import matplotlib.pyplot as plt


def moving_average(values, window):
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_all_together(agent_names, env_name, n_runs, title):
    fig = plt.figure(title)
    min_run_len = math.inf
    ys = []

    for a in agent_names:
        max_run_len = 0
        runs = []
        for i in range(n_runs):
            run_x, run_y = ts2xy(load_results('log_{}_{}_{}/'.format(a, env_name, i)), 'timesteps')
            runs.append((run_x, run_y))
            if max_run_len < run_x.shape[0]:
                max_run_len = run_x.shape[0]
            print('agent: ', a)
            print('run_x: ', run_x.shape)
            print('run_y: ', run_y.shape)

            #if run_x.shape[0] < min_run_len:
            #    min_run_len = run_x.shape[0]
        y = np.full((max_run_len, n_runs), np.nan)

        ix = 0
        for (_, run_y) in runs:
            y[:run_y.shape[0], ix] = run_y
            ix = ix+1
        
        ys.append(y)

    i = 0
    for a in agent_names:
        y = ys[i]
        i = i + 1
        #y = y[:min_run_len,:]
        x = np.arange(y.shape[0])
        print('x shape: ', x.shape)
        print('y shape: ', y.shape)

        mean = np.mean(y, axis=1)
        std = np.std(y, axis=1)
        print('mean shape', mean.shape)
        print('std shape', std.shape)

        mean = moving_average(mean, window=50)
        std = moving_average(std, window=50)
        x = x[len(x) - len(mean):]
        x = x * (1e5 / len(x))
        plt.plot(x, mean, label=a.upper())
        plt.fill_between(x, mean-std, mean+std, alpha=0.3)

    plt.grid(color='grey', linestyle='-', linewidth=1)
    plt.xlabel('Timesteps')
    plt.ylabel('Reward')
    plt.title(title)
    plt.legend()
    plt.show()



#env_name = 'LunarLanderContinuous-v2'
#env_name = 'BipedalWalker-v2'
env_name = 'MountainCarContinuous-v0'
n_runs = 3

agent_names = ['sac','a2c','ppo']
plot_all_together(agent_names, env_name, n_runs, env_name)