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

    #TODO make this a more general procedure to take in an environment name
    env = DummyVecEnv([lambda: SimpleCorridor(cfg)])
    #load our trained agent
    model = PPO2.load(args.pretrained)

    obs = env.reset()
    for i in range(args.num_rollout_steps):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        #env.render()
    env.render(save=True)