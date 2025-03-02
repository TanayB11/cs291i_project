import gymnasium as gym
from stable_baselines3 import PPO
from tqdm import tqdm
from gymnasium.wrappers import RecordVideo
import numpy as np

class CustomAntEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.step_count = 0
        self.max_steps = 100

        self.ctrl_cost_weight = 0.8

        self.height_threshold = 1.0
        self.height_bonus = 5

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        reward = 0
        if terminated:
            return obs, reward, terminated, truncated, info

        torso_height = obs[0]
        if torso_height > self.height_threshold:
            reward += self.height_bonus
        
        reward += 1 # staying healthy
        reward += info['reward_contact'] # <= 0
        reward += info['reward_ctrl'] # <= 0

        # Increment step counter
        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.step_count = 0
        return self.env.reset(**kwargs)

# create environments, train the environment
base_env = gym.make("Ant-v5", healthy_z_range=(0.2, 2.5))
env = CustomAntEnv(base_env)
model = PPO("MlpPolicy", env)
model.learn(total_timesteps=10_000_000)
env.close()
model.save("ant_ppo_model")

# Load the trained model
base_record_env = gym.make("Ant-v5", render_mode="rgb_array", healthy_z_range=(0.2, 2.5))
model = PPO.load("ant_ppo_model.zip")
record_env = CustomAntEnv(base_record_env)
record_env = RecordVideo(record_env, video_folder="./out_10m", episode_trigger=lambda x: True)

for episode in range(10):
    obs, _ = record_env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = record_env.step(action)
        done = terminated or truncated

record_env.close()