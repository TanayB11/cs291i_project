# MuJoCo Environment Demo

This project demonstrates how to use MuJoCo physics engine with Gymnasium for reinforcement learning, featuring the Ant-v5 environment.

## Features

- MuJoCo environment visualization with Gymnasium
- Pre-trained PPO model demonstration
- Frame extraction and collage creation tools

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   cd YOUR_REPO_NAME
   ```

2. Install dependencies:
   ```bash
   pip install gymnasium stable-baselines3 opencv-python numpy
   ```

3. MuJoCo is included with Gymnasium, no additional installation required.

## Usage

### Running the MuJoCo Demo

```bash
python gym_mujoco_demo.py
```

This will run a simulation of the Ant-v5 environment using a pre-trained PPO model.

### Creating Frame Collages

```bash
python get_frames.py
```

This script extracts frames from simulation videos and creates a collage.

## Project Structure

- `gym_mujoco_demo.py`: Main demo script for running the MuJoCo environment
- `get_frames.py`: Utility for extracting frames and creating collages
- `ant_ppo_model.zip`: Pre-trained PPO model for the Ant-v5 environment

## Customization

Modify the environment parameters in `gym_mujoco_demo.py` to experiment with different:
- Environments (default: Ant-v5)
- Rendering modes
- Model parameters

## License

[Your License Here]