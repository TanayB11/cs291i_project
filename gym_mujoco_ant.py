import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
import argparse
import numpy as np
import os
import wandb
import datetime

class CustomAntEnv(gym.Wrapper):
    def __init__(self, env, env_step_fn):
        super().__init__(env)
        self.step_count = 0
        self.max_steps = 100

        self.healthy_reward = 1
        self.forward_reward_weight = 1
        self.ctrl_cost_weight = 0.5
        self.contact_cost_weight = 5e-4

        self.height_threshold = 1.0
        self.height_bonus = 10 

        self.gpt_reward = 0
        self.gpt_reward_weight = 2
        
        # defines environment step
        self.env_step_fn = env_step_fn

    def reset(self, **kwargs):
        self.step_count = 0
        return self.env.reset(**kwargs)


def default_step(self, action):
    """
    Default step function for the environment
    """
    obs, reward, terminated, truncated, info = self.env_step_fn(action)
    return obs, reward, terminated, truncated, info


class CheckpointCallback(CheckpointCallback):
    def __init__(self, save_freq, save_path, name_prefix):
        super().__init__(save_freq=save_freq, save_path=save_path, name_prefix=name_prefix)
        
    def _on_step(self):
        if (self.n_calls) % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.n_calls}_steps")
            self.model.save(path)
            print(f"Saved model to {path}")
        return True


class TrajectoryRewardCallback(BaseCallback):
    """Callback for logging trajectory rewards to wandb
    
    This callback collects reward information for each trajectory/episode and logs
    it to wandb for visualization.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.episode_count = 0
    
    def _on_step(self):
        # Get info from the last step
        reward = self.locals['rewards'][-1]
        self.current_episode_reward += reward
        
        # Check if episode is done
        if self.locals['dones'][-1]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_count += 1
            
            # Log to wandb
            wandb.log({
                'episode_reward': self.current_episode_reward,
                'episode': self.episode_count,
                'timestep': self.num_timesteps
            })
            
            # Reset for next episode
            self.current_episode_reward = 0
            
            # Periodically log average reward over last 1000 episodes
            if len(self.episode_rewards) >= 1000:
                avg_reward = np.mean(self.episode_rewards[-1000:])
                wandb.log({
                    'avg_episode_reward_last_1000': avg_reward,
                    'episode': self.episode_count,
                    'timestep': self.num_timesteps
                })
        
        return True

if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Train an Ant-v5 agent using PPO')
    parser.add_argument('--model-dir', type=str, default=None, help='Directory containing the model to load')
    parser.add_argument('--model-name', type=str, default='ant_ppo_model.zip', help='Name of the model file to load')
    parser.add_argument('--output-dir', type=str, default='./ant_out', help='Directory to save outputs')
    parser.add_argument('--train-steps', type=int, default=5e6, help='Number of timesteps to train')
    parser.add_argument('--save-freq', type=int, default=10_000, help='Frequency to save model checkpoints')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create environment
    env = CustomAntEnv(gym.make("Ant-v5"), default_step)

    # Initialize wandb with API key from .env
    run_name = f"ant-ppo-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    wandb.init(project="ant-ppo", name=run_name)
    
    # Log environment config
    wandb.config.update({
        "env_id": "Ant-v5",
        "healthy_reward": env.healthy_reward,
        "forward_reward_weight": env.forward_reward_weight,
        "ctrl_cost_weight": env.ctrl_cost_weight,
        "contact_cost_weight": env.contact_cost_weight,
        "height_threshold": env.height_threshold,
        "height_bonus": env.height_bonus,
        "gpt_reward_weight": env.gpt_reward_weight,
        "max_steps": env.max_steps,
        "train_steps": int(args.train_steps)
    })

    # Load the model if specified
    if args.model_dir and args.model_name and os.path.exists(os.path.join(args.model_dir, args.model_name)):
        model_path = os.path.join(args.model_dir, args.model_name)
        print(f"Loading model from {model_path}")
        model = PPO.load(model_path, env=env)
    else:
        print("Creating a new model")
        model = PPO("MlpPolicy", env, verbose=1)
    
    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=args.output_dir,
        name_prefix='ant'
    )
    reward_callback = TrajectoryRewardCallback()
    callbacks = CallbackList([checkpoint_callback, reward_callback])
    
    # Train the model
    print(f"Training for {args.train_steps} steps")
    model.learn(total_timesteps=args.train_steps, callback=callbacks, reset_num_timesteps=False)
    
    # Log final model performance
    if hasattr(model, "ep_info_buffer") and len(model.ep_info_buffer) > 0:
        final_reward = np.mean([ep_info["r"] for ep_info in model.ep_info_buffer])
        wandb.log({"final_reward": final_reward})
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "ant_ppo_final_model")
    model.save(final_model_path)
    print(f"Saved final model to {final_model_path}")
    
    wandb.finish()
    env.close()