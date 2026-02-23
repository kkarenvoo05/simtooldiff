# Installation

## Main Environment

We use Isaac Gym for policy learning in simulation, which requires Python 3.8.

```
# Create a new conda environment for the Isaac Gym environment
conda create -n simtoolreal_env python=3.8  # isaacgym requires Python 3.8
conda activate simtoolreal_env

# Misc
pip install \
  torch torchvision \
  numpy scipy matplotlib networkx \
  gym pyrender trimesh urdfpy \
  hydra-core wandb tensorboard tensorboardx \
  opencv-python imageio pillow transforms3d \
  ipykernel \
  requests pyopenssl \
  warp-lang ninja pyvirtualdisplay

# More dependencies
pip install pytorch3d "imageio[ffmpeg]" yourdfpy viser pytorch_kinematics mujoco ruff tyro

# Isaacgym stubs: https://x.com/QinYuzhe/status/1800288199136416178
pip install isaacgym-stubs --upgrade

# Download the Isaac Gym Preview 4 release from https://developer.nvidia.com/isaac-gym
cd isaacgym/python
pip install -e .
pip install numpy==1.23.0  # isaacgym does not support numpy 1.24+

# Install this repo's rl_games
cd <this repo>/rl_games
pip install -e .

# Install this repo
cd <this repo>
pip install -e .
```

## Sim2Real Env

For real-world deployment, we use need ROS1 Noetic. This can either be installed globally or using RoboStack.

### Global ROS

Follow the instructions at https://wiki.ros.org/noetic/Installation/Ubuntu to install ROS1 Noetic globally. Then you will need to source the ROS environment variables in your shell:

```
source /opt/ros/noetic/setup.bash
```

Now, you can simply use this global ROS setup with the `simtoolreal_env` conda environment above.

You may need to run:

```
pip install rospkg
```

### RoboStack

Note that the RoboStack option requires a separate conda environment because it requires Python 3.11+. Follow the instructions at https://robostack.github.io/noetic.html to create a conda environment with ROS1 Noetic. Then follow the installation instructions above for the main environment (skip the conda environment creation and Isaac Gym installation).