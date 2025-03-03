# traditional training if you want to use checkpoint for pre-trained model

import gymnasium as gym
from stable_baselines3 import PPO
import time
import keyboard
from stable_baselines3.common.callbacks import CheckpointCallback

env = gym.make('CarRacing-v3', render_mode="human")

checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./checkpoints/',
                                         name_prefix='car_racing')

model = PPO("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000, callback=checkpoint_callback)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    if done:
        obs = env.reset()

env.close()