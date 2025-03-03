import gymnasium as gym
from stable_baselines3 import PPO
import argparse
from train_hack import HackyCarRacing

def main(checkpoint_path):
    # regular environment
    # env = gym.make('CarRacing-v3', render_mode="human")
    
    # hacky environment
    env = HackyCarRacing(gym.make('CarRacing-v3', render_mode="human"))

    model = PPO.load(checkpoint_path, env=env)
    obs, info = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        if done or truncated:
            break
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a checkpoint to assess performance.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint file.")
    args = parser.parse_args()
    main(args.checkpoint)
