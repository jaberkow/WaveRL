"""
This script is rolls out a trained agent.

Currently it takes in several command line arguments:

-n:  The number of timesteps to rollout for (default is set in config.yml)
-i:  A path specifying a pretrained agent .pkl file to load and rollout

It then builds the environment, policy network, rolls out the agent and produces a visualization in test.png.
"""

import gym

#load the stable_baselines functions
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

#currently this just loads the simple corridor environment, more will be added later
from simple_environment import SimpleCorridor

#other utilities
import yaml
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n',dest='num_rollout_steps',
        help='The number of rollout steps', default=-1,type=int)
    parser.add_argument('-i', dest='pretrained',
        help='Path to a pretrained agent to rollout',default='', type=str)
    args = parser.parse_args()
    
    #load the config variables, currently assuming the config file is 
    #in a fixed relative path to this script
    with open('../configs/config.yml','r') as yamlfile:
        cfg=yaml.load(yamlfile)

    #check if we override the number of rollout steps
    if args.num_rollout_steps >0:
        rollout_steps = args.num_rollout_steps
    else:
        rollout_steps = cfg['num_rollout_steps']

    #TODO make this a more general procedure to take in an environment name
    env = DummyVecEnv([lambda: SimpleCorridor(cfg)])
    #Make sure a proper pretrained agent file was passed
    assert args.pretrained.endswith('.pkl') and os.path.isfile(args.pretrained), "The pretrained agent must be a valid path to a .pkl file"

    #load our trained agent
    model = PPO2.load(args.pretrained)

    obs = env.reset()
    for i in range(rollout_steps):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        #env.render()
    env.render(save=True)