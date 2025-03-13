import gymnasium as gym
from stable_baselines3 import PPO
from tqdm import tqdm
from gymnasium.wrappers import RecordVideo
import argparse
import numpy as np
import os
from dotenv import load_dotenv
import re
from recordingCallback import VideoRecordingCallback

class CustomCheetahGPTEnv(gym.Wrapper):    
    def __init__(self, env, alpha=1):
        super().__init__(env)
        self.step_count = 0
        self.max_steps = 100
        self.gpt_score = 0.5

        self.alpha = alpha
        self.bad_behavior_cutoff = -5
        self.good_behavior_cutoff = 1
        self.current_episode_true_reward = 0
        self.current_episode_artificial_reward = 0
        
    def asym_logit(self, gpt_score):
        if gpt_score < 0.5:
            factor = -np.exp(self.bad_behavior_cutoff) / (np.exp(self.bad_behavior_cutoff) - 1)
            return np.log((gpt_score + factor) / (factor + 1 - gpt_score))
        else:
            factor = 1.0 / (np.exp(self.good_behavior_cutoff) - 1)
            return np.log((gpt_score + factor) / (factor + 1 - gpt_score))
    
    def set_gpt_score(self, score):
        self.gpt_score = score
    
    def step(self, action):
        obs, true_reward, terminated, truncated, info = self.env.step(action)
        
        forward_reward = info.get("reward_forward", 0.0)  # Forward reward component
        ctrl_cost = info.get("reward_ctrl", 0.0)      # Control cost component
        bfoot_velocities = (abs(obs[4]), abs(obs[13]))
        artificial_reward = forward_reward + ctrl_cost
        vlm_reward = self.asym_logit(self.gpt_score)
        artificial_reward += self.alpha*vlm_reward
        
        self.current_episode_artificial_reward += artificial_reward
        self.current_episode_true_reward += true_reward
        
        # Increment step counter
        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True

        return obs, artificial_reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.step_count = 0
        self.current_episode_true_reward = 0
        self.current_episode_artificial_reward = 0
        return self.env.reset(**kwargs)
    

if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Train a HalfCheetah-v5 agent using PPO')
    parser.add_argument('--run-gpt-eval', action='store_true', default=False, help='Enable GPT evaluation during training')
    parser.add_argument('--model-dir', type=str, default=None, help='Directory containing the model to load')
    parser.add_argument('--output-dir', type=str, default='./out/cheetah_hack_terminates', help='Directory to save outputs')
    parser.add_argument('--model-name', type=str, default='cheetah_ppo_model.zip', help='Name of the model file to load')
    args = parser.parse_args()
    
    load_dotenv()

    # create environments, train the environment
    base_env = gym.make("HalfCheetah-v5")
    env = CustomCheetahGPTEnv(base_env)
    
    # Load the model from the specified directory if provided
    if args.model_dir:
        if args.model_name: model_path = os.path.join(args.model_dir, args.model_name)
        else:
            # Find the latest model in the directory (with highest step count)
            model_files = [f for f in os.listdir(args.model_dir) if f.endswith('_steps.zip')]
            if model_files:
                model_files.sort(key=lambda x: int(re.search(r'(\d+)_steps', x).group(1)) if re.search(r'(\d+)_steps', x) else 0, reverse=True)
                model_path = os.path.join(args.model_dir, model_files[0])
            else:
                model_path = None
        
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            model = PPO.load(model_path, env=env)
        else:
            print(f"Model not found, creating a new model")
            model = PPO("MlpPolicy", env, verbose=1, device='cpu')
    else:
        print("No model directory specified, creating a new model")
        model = PPO("MlpPolicy", env, verbose=1, device='cpu')

    callback = VideoRecordingCallback(
        save_freq=10000,
        root_folder=args.output_dir,
        name_prefix='HalfCheetah-v5',
        run_gpt_eval=args.run_gpt_eval,
        env=env,
        env_class=CustomCheetahGPTEnv
    )
    
    total_steps_to_train = 500_000    
    model.learn(total_timesteps=total_steps_to_train, callback=callback, reset_num_timesteps=False)
    env.close()
    model.save("cheetah_ppo_model")