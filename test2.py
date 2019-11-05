import os

import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines import results_plotter
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import A2C
from stable_baselines.common.noise import AdaptiveParamNoiseSpec, NormalActionNoise
from stable_baselines.common.vec_env import DummyVecEnv
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

# Create log dir
log_dir = "logging/"
os.makedirs(log_dir, exist_ok=True)

# create environment
env = gym.make('Pendulum-v0')
env = Monitor(env, log_dir, allow_early_resets=True)
env = DummyVecEnv([lambda: env])

# create and train our agent
model = A2C(MlpPolicy, env, verbose=0)
model.learn(total_timesteps=int(1e5))

# plot results
plt.figure()
results_plotter.plot_results(["./logging"], None, results_plotter.X_EPISODES, "Pendulum-v0")
plt.savefig("test.png")
