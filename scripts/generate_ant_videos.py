#!/usr/bin/env python3
import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import argparse
from tqdm import tqdm
import imageio
from gymnasium.wrappers import RecordVideo
import sys

# Add parent directory to path so we can import from gym_mujoco_demo
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import the CustomAntEnv from your gym_mujoco_demo.py
from gym_mujoco_demo import CustomAntEnv


def create_video_from_frames(frames, output_path, fps=30):
    """Create a video from a list of frames."""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write frames to video file
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"Video saved to {output_path}")


def rollout_trajectory(model, env, video_length, video_path):
    """Roll out a single trajectory and save it as a video."""
    # Reset the environment
    obs, _ = env.reset()
    frames = []
    
    # Roll out the trajectory
    for _ in range(video_length):
        # Get the action from the model
        action, _ = model.predict(obs, deterministic=True)
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the frame
        frame = env.render()
        frames.append(frame)
        
        # Check if the episode is done
        if terminated or truncated:
            break
    
    # Create a video from the frames
    create_video_from_frames(frames, video_path)
    
    return frames


def main():
    parser = argparse.ArgumentParser(description='Generate videos from a trained Ant-v5 model')
    parser.add_argument('--model-path', type=str, 
                        default='../out/ant_hack/checkpoints/ant_5200000_steps.zip',
                        help='Path to the trained model')
    parser.add_argument('--output-dir', type=str, 
                        default='../out/ant_hack_videos',
                        help='Directory to save output videos')
    parser.add_argument('--num-videos', type=int, default=10,
                        help='Number of videos to generate')
    parser.add_argument('--video-length', type=int, default=500,
                        help='Length of each video in frames')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create the Ant environment with the same wrapper used during training
    base_env = gym.make("Ant-v5", render_mode="rgb_array")
    env = CustomAntEnv(base_env)
    
    # Load the trained model
    print(f"Loading model from {args.model_path}")
    model = PPO.load(args.model_path, env=env)
    
    # Generate videos
    print(f"Generating {args.num_videos} videos...")
    for i in tqdm(range(args.num_videos)):
        video_path = os.path.join(args.output_dir, f"ant_trajectory_{i}.mp4")
        frames = rollout_trajectory(model, env, args.video_length, video_path)
    
    print(f"All videos generated successfully in {args.output_dir}")
    env.close()


if __name__ == "__main__":
    main()
