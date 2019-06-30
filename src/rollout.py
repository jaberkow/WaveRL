"""
This script produces vi

Currently it takes in several command line arguments:

-n:  The number of timesteps to rollout for (default is set in config.yml)
-i:  A path specifying a pretrained agent .pkl file to load and rollout
-f:  A  path specifying the name of the output file to record the rollout

It then builds the environment, policy network, rolls out the agent and records the rollout in npz file.
"""

import gym

# Load the stable_baselines functions
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from active_damping_env import VibratingBridge

# Other utilities
import yaml
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n',dest='num_rollout_steps',
        help='The number of rollout steps', default=-1,type=int)
    parser.add_argument('-i', dest='pretrained',
        help='Path to a pretrained agent to rollout',default='', type=str)
    parser.add_argument('-f',dest='output_filename',
        help='Name of output file',default='output',type=str)
    args = parser.parse_args()
    
    # Load the config variables, currently assuming the config file is 
    # in a fixed relative path to this script
    with open('../configs/config.yml','r') as yamlfile:
        cfg=yaml.load(yamlfile)

    # Check if we override the number of rollout steps
    if args.num_rollout_steps >0:
        rollout_steps = args.num_rollout_steps
    else:
        rollout_steps = cfg['num_rollout_steps']

    # Setup the environment
    env=DummyVecEnv([lambda: VibratingBridge(cfg)])
    # Make sure a proper pretrained agent file was passed
    assert args.pretrained.endswith('.pkl') and os.path.isfile(args.pretrained), "The pretrained agent must be a valid path to a .pkl file"

    # Load our trained agent
    model = PPO2.load(args.pretrained,env=env)

    obs = env.reset()
    for i in range(rollout_steps):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
    env.render(fname=args.output_filename)