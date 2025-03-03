import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
import argparse
from train_hack import HackyCarRacing

def main(checkpoint_path):
    # Wrap the environment so it records a video
    env = gym.make("CarRacing-v3", render_mode="rgb_array_list")
    env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda e: True)

    model = PPO.load(checkpoint_path, env=env)
    obs, info = env.reset()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            break

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record checkpoint as an MP4.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint file.")
    args = parser.parse_args()
    main(args.checkpoint)
