"""
This script is the main training script.

Currently it takes in several command line arguments:

-tb:  A path where the tensorboard information will be saved (reward, etc)
-n:  The number of timesteps to train for (default is set in config.yml)
-i:  A path specifying a pretrained agent .pkl file to load and continue training
-m:  A string that will form the filename of the saved file
-lr:  A float representing the learning rate for the PPO2 algorithm (default is set in config.yml)

It then builds the environment, policy network, trains the agent, and saves the trained model.
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

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-tb', dest='tensorboard_log_dir',
		help='Tensorboard log dir', default='tensorboard_log', type=str)
	parser.add_argument('-n',dest='num_learning_steps',
		help='Overwrite the number of learning steps', default=-1,type=int)
	parser.add_argument('-i', dest='pretrained',
		help='Path to a pretrained agent to continue training',default='', type=str)
	parser.add_argument('-m',dest='model_name',
		help='Save the trained model here',default='trained_agents/trained_model',type=str)
	parser.add_argument('-lr',dest='learning_rate_val',
		help='Overwrite the learning rate',default=-1.0,type=float)
	args = parser.parse_args()

	# Make sure we find where the config file is
	CWD_PATH = os.getcwd()
	config_path = os.path.join(CWD_PATH,'configs/config.yml')
	with open(config_path, 'r') as ymlfile:
		cfg = yaml.load(ymlfile,Loader=yaml.FullLoader)

	# Setup the environment
	env=DummyVecEnv([lambda: VibratingBridge(cfg)])
	# Do we overwrite the learning rate
	if args.learning_rate_val >0:
		learning_rate = args.learning_rate_val
	else:
		learning_rate = cfg['learning_rate_val']

	# If we're using pretrained model make sure it's in the right format
	if args.pretrained !='':
		assert args.pretrained.endswith('.pkl') and os.path.isfile(args.pretrained), "The pretrained agent must be a valid path to a .pkl file"
		if args.pretrained.endswith('.pkl') and os.path.isfile(args.pretrained):
			model = PPO2.load(args.pretrained,env=env,verbose=1,tensorboard_log=args.tensorboard_log_dir)
	else:
		model = PPO2(MlpPolicy, env=env, verbose=0,tensorboard_log=args.tensorboard_log_dir,learning_rate=learning_rate)

	# Set the number of training steps
	if args.num_learning_steps >0:
		steps_to_train = args.num_learning_steps
	else:
		steps_to_train = cfg['num_learning_steps']
	model.learn(total_timesteps=steps_to_train) # Train the model
	model.save(args.model_name) # Save the model






