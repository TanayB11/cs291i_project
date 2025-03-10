#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_metrics(csv_path, title):
    """
    Plot reward, smoothed reward, GPT reward, and smoothed GPT reward
    with a moving average window of 100 time steps.
    
    Args:
        csv_path (str): Path to the CSV file
        title (str): Title for the plot
    """
    # Read CSV data
    df = pd.read_csv(csv_path)
    
    # Calculate summary statistics
    reward_mean = df['avg_reward'].mean()
    reward_var = df['avg_reward'].var()
    gpt_mean = df['latest_gpt_score'].mean()
    gpt_var = df['latest_gpt_score'].var()
    
    print(f"\nSummary Statistics for {title}:")
    print(f"Reward Mean: {reward_mean:.2f}, Variance: {reward_var:.2f}")
    print(f"GPT Reward Mean: {gpt_mean:.2f}, Variance: {gpt_var:.2f}")
    
    # Count total data points
    total_points = len(df)
    print(f"Total data points: {total_points}")
    
    # Determine appropriate window size (use smaller window if data points are limited)
    window_size = min(100, max(5, total_points // 10))
    print(f"Using window size of {window_size} for smoothing")
    
    # Apply moving average smoothing with adjusted window size
    df['smoothed_reward'] = df['avg_reward'].rolling(window=window_size, min_periods=1).mean()
    df['smoothed_gpt_score'] = df['latest_gpt_score'].rolling(window=window_size, min_periods=1).mean()
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    
    # Plot reward and smoothed reward on left y-axis
    ax1.plot(df['timestep'], df['avg_reward'], 'b-', alpha=0.3, linewidth=1, label='Reward')
    ax1.plot(df['timestep'], df['smoothed_reward'], 'b-', linewidth=2.5, label=f'Smoothed Reward (MA-{window_size})')
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Reward', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Plot GPT score and smoothed GPT score on right y-axis
    ax2.plot(df['timestep'], df['latest_gpt_score'], 'r-', alpha=0.3, linewidth=1, label='GPT Reward')
    ax2.plot(df['timestep'], df['smoothed_gpt_score'], 'r-', linewidth=2.5, label=f'Smoothed GPT Reward (MA-{window_size})')
    ax2.set_ylabel('GPT Reward', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Create combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    # Set title and layout with statistics
    stats_title = f"{title}\nReward: μ={reward_mean:.2f}, σ²={reward_var:.2f} | GPT Reward: μ={gpt_mean:.2f}, σ²={gpt_var:.2f}"
    plt.title(stats_title)
    plt.tight_layout()
    
    # Save figure with relative path
    # Extract directory from csv_path but make it relative
    base_dir = os.path.dirname(os.path.dirname(csv_path))
    save_path = os.path.join(base_dir, f"plot_{title.replace(' ', '_').lower()}.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    return fig

def main():
    # Plot first CSV with relative paths
    gpt_csv = "../out/ant_hack_gpt/metrics/training_metrics.csv"
    fig1 = plot_metrics(gpt_csv, "Ant Hack GPT Training Metrics")
    
    # Plot second CSV with relative paths
    regular_csv = "../out/ant_hack/metrics/training_metrics.csv"
    fig2 = plot_metrics(regular_csv, "Ant Hack Training Metrics")
    
    # Show plots
    plt.show()

if __name__ == "__main__":
    main()
