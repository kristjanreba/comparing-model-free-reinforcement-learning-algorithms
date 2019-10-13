import gym

from stable_baselines import A2C, SAC, PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)



# Define and train a model in one line of code !
trained_model = A2C('MlpPolicy', 'CartPole-v1').learn(total_timesteps=10000)
# you can then access the gym env using trained_model.get_env()
trained_model.get_env()
print('Finished 1')




env = gym.make('CartPole-v1')
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

model = A2C(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

env.close()

print('Finished 2')
