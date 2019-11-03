import gym

from stable_baselines import results_plotter
from stable_baselines import A2C, SAC, PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.sac.policies import MlpPolicy as SacMlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)


def render_env(env, model):
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
    env.close()


if __name__ == '__main__':

    env = gym.make('Pendulum-v0')
    env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

    model_A2C = A2C(MlpPolicy, env, verbose=1)
    model_A2C.learn(total_timesteps=10000)

    model_SAC = SAC(SacMlpPolicy, env, verbose=1)
    model_SAC.learn(total_timesteps=10000)

    model_PPO2 = PPO2(MlpPolicy, env, verbose=1)
    model_PPO2.learn(total_timesteps=10000)

    render_env(env, model_A2C)
    render_env(env, model_SAC)
    render_env(env, model_PPO2)

    results_plotter.main()

