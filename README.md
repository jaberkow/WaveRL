# WaveRL
A package for training RL agents to perform active damping on a model of a vibrating bridge.

## Contents
* complex_envs/ :  This folder contains the code for the environments as well as scripts for training and rolling out agents
	* train.py :  This script trains an agent (see example below)
	* rollout.py : This script rolls out a trained agent (see example below)
	* simple_environment.py :  This in an implementation of the simple corridor environment
	* finite_diff_wave.py : This is a class definition for a simulator of one dimensional wave equation with finite difference methods.
* configs/ 
	* config.yml : This file holds the default parameters for the scripts and environments
* install_stable_requirements.sh : a shell script for installing all the necessary packages
* conda_requirements_baseline.yaml : A specification of the conda environment

## Setup
First, [Install Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)

Then update and install the following system packages:

### Ubuntu Instructions:
```
$ sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev openmpi-bin mpich lam-runtime
```
### Mac OS X Instructions:
Installing the necessary C-libraries is easiest with [Homebrew](https://brew.sh/), so install this first:
```
$ /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```
Then install cmake and openmpi.
```
$ brew install cmake openmpi
```
### Setting up the conda environment

Clone the repository and change into the repo folder
```
$ git clone https://github.com/jaberkow/WaveRL.git
$ cd WaveRL
```

Make a conda environment and activate it:

```
$ conda create -n WaveEnv python=3.6
$ conda activate WaveEnv
```
Install the packages from `requirements.txt`

```
$ pip install -r requirements.txt
```

## Training an agent

To train an agent for 1000 timesteps and save it as 'my_first_agent.pkl', navigate to the `complex_envs` folder and run the following command:

```
$ cd complex_envs
$ python train.py -n 1000 -m my_first_agent 
```
This command will also produce a TensorBoard folder at `./tensorboard_log` that can be visualized with

```
$ tensorboard --logdir tensorboard_log/
```
## Rolling out a trained agent

To rollout a trained agent that is stored at 'complex_envs/my_first_agent.pkl' for 9 steps, run the following command from the same folder:

```
$ python rollout.py -n 9 -i my_first_agent.pkl
```
