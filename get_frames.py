import cv2
import os
import numpy as np
from pathlib import Path
from math import ceil, sqrt

def create_video_collage(video_path, output_path, max_frames=25, frame_interval=1):
    """
    Create a collage of frames from a video file.
    
    Args:
        video_path (str): Path to the input video file
        output_path (str): Path to save the output collage image
        max_frames (int): Maximum number of frames to include in the collage
        frame_interval (int): Extract every nth frame (default: 1)
    """
    # Open the video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")
    
    # Get video properties
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate grid size
    grid_size = ceil(sqrt(max_frames))
    frames = []
    frame_count = 0
    saved_count = 0
    
    while saved_count < max_frames:
        success, frame = video.read()
        if not success:
            break
            
        if frame_count % frame_interval == 0:
            frames.append(frame)
            saved_count += 1
            
        frame_count += 1
    
    video.release()
    
    if not frames:
        raise ValueError("No frames were extracted from the video")
    
    # Resize frames to a consistent size
    target_size = (300, 300)  # You can adjust this size
    resized_frames = [cv2.resize(frame, target_size) for frame in frames]
    
    # Create the collage
    rows = cols = grid_size
    collage = np.zeros((rows * target_size[0], cols * target_size[1], 3), dtype=np.uint8)
    
    for idx, frame in enumerate(resized_frames):
        i, j = divmod(idx, cols)
        if i >= rows:
            break
        y_start = i * target_size[0]
        y_end = (i + 1) * target_size[0]
        x_start = j * target_size[1]
        x_end = (j + 1) * target_size[1]
        collage[y_start:y_end, x_start:x_end] = frame
    
    # Save the collage
    cv2.imwrite(output_path, collage)
    print(f"Created collage with {len(frames)} frames from {total_frames} total frames")

if __name__ == "__main__":
    # Example usage
    video_path = "out_10m/rl-video-episode-1.mp4"
    output_path = "./frames/video_collage.jpg"
    create_video_collage(video_path, output_path, max_frames=49, frame_interval=2)