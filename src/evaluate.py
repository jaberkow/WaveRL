"""
This script takes in a trained agent, performs multiple rollouts, and records
the number of steps it takes the agent to dissipate the energy to below a threshold

Currently it takes in several command line arguments:

-n:  The number of timesteps to rollout for (default is set in config.yml)
-i:  A path specifying a pretrained agent .pkl file to load and evaluate
-f:  A  path specifying the name of the output file to record the evaluation results
-t:  The energy threshold, 0.1 means a threshold of 10% the average energy during
    the equilibriation phase
-r: The number of evaluation repeats to perform (default is set in config.yml)

"""
import sys
sys.path.append('..')
import gym

# Load the stable_baselines functions
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from environments.active_damping_env import VibratingBridge

# Other utilities
import yaml
import argparse
import os
import numpy as np

def steps_to_threshold(data,threshold):
    """
    Takes in a numpy archive representing a rollout and returns how many
    damping steps it took the agent to get the energy under threshold. If
    the agent never gets the energy below threshold it returns the total number
    of damping steps.

    Inputs:
        data:  Numpy archive, must have energy_array and code_array keys
        threshold:  Float between 0 and 1, the relative energy threshold
    Outputs:
        damping_steps:  How many steps it took the agent to get energy below
            threshold 
    """
    energy_array = data['energy_array']
    code_array = data['code_array']
    total_steps = np.size(energy_array)
    damping_steps = 0
    for i in range(total_steps):
        if code_array[i]==2:
            if energy_array[i] < threshold:
                return damping_steps
            else:
                damping_steps += 1
    return damping_steps

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n',dest='num_rollout_steps',
        help='The number of rollout steps', default=-1,type=int)
    parser.add_argument('-i', dest='pretrained',
        help='Path to a pretrained agent to rollout',default='', type=str)
    parser.add_argument('-f',dest='output_filename',
        help='Name of output file',default='trained_agents/output',type=str)
    parser.add_argument('-t',dest='threshold',
		help='The energy threshold',default=-1.0,type=float)
    parser.add_argument('-r',dest='evaluation_reps',
		help='How many evaluation repeats to do',default=-1,type=int)
    args = parser.parse_args()

    # Make sure we find where the config file is
    CWD_PATH = os.getcwd()
    config_path = os.path.join(CWD_PATH,'configs/config.yml')
    with open(config_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile,Loader=yaml.FullLoader)

    # Check if we override the number of evaluation repeats
    if args.evaluation_reps >0:
        evaluation_repeats = args.evaluation_reps
    else:
        evaluation_repeats = cfg['evaluation_reps']

    # Check if we override the number of rollout steps
    if args.num_rollout_steps >0:
        rollout_steps = args.num_rollout_steps
    else:
        rollout_steps = cfg['num_rollout_steps']

    # Do we overwrite the threshold
    if args.threshold >0:
        threshold = args.threshold
    else:
        threshold = cfg['threshold']

    # Setup the environment
    env=DummyVecEnv([lambda: VibratingBridge(cfg)])
    # Make sure a proper pretrained agent file was passed
    assert args.pretrained.endswith('.pkl') and os.path.isfile(args.pretrained), "The pretrained agent must be a valid path to a .pkl file"

    # Load our trained agent
    model = PPO2.load(args.pretrained,env=env)

    steps_list = []
    for rep in range(evaluation_repeats):
        obs = env.reset()
        for i in range(rollout_steps):
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
        env.render(fname='eval_temp')
        data = np.load('eval_temp.npz')
        steps_result = steps_to_threshold(data,threshold)
        steps_list.append(steps_result)
    os.remove('eval_temp.npz')
    np.save(args.output_filename,steps_list)
