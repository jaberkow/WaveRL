# WaveRL
A package for training RL agents to perform active damping on a model of a oscillating bridge.

## Contents
* complex_envs/ :  This folder contains the code for the environments as well as scripts for training and rolling out agents
	* train.py :  This script trains an agent (see example below)
	* rollout.py : This script rolls out a trained agent (see example below)
	* visualize.py : This script produces visualizations of a rolled out agent (see example below)
	* finite_diff_wave.py : This is a class definition for a simulator of one dimensional wave equation with finite difference methods.
	* active_damping_env.py : This is a class definition for an OpenAI gym environment simulating an oscillating bridge
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

## About the environment

This package simulates an oscillating bridge by modelling it with the one-dimensional [wave equation](https://en.wikipedia.org/wiki/Wave_equation), which is simulated using a [finite difference solver](https://en.wikipedia.org/wiki/Finite_difference_method).  The action space of the environment represents pistons that apply a force to actively dampen vibrations in the bridge.  The reward signal is inversely proportional to the interal energy of the system.  One episode of the environment involves 3 phases:  1) A "warmup phase" where an external force is applied to the system to cause oscillations 2) An "equilibriation" phase where the oscillations settle in to stable patterns and 3) A dampening phase where the agent attempts to dampen the oscillations.

## Training an agent

To train an agent for 1000 timesteps on the vibrating bridge environment and save it as `my_first_agent.pkl`, navigate to the `complex_envs` folder and run the following command:

```
$ cd complex_envs
$ python train.py -n 1000 -m my_first_agent
```
This command will also produce a TensorBoard folder at `./tensorboard_log` that can be visualized with

```
$ tensorboard --logdir tensorboard_log/
```
## Rolling out a trained agent

To rollout a trained agent that is stored at `complex_envs/my_first_agent.pkl` for 60 steps, run the following command from the same folder:

```
$ python rollout.py -n 60 -i my_first_agent.pkl
```

The rollout will be saved as `output.npz`, which can be changed by passing `-f <filename>` to the above command.  Note that the trajectories produced will have length equal to the number of rollout steps + the number of warmup steps + the number of equilibriation steps (values of which are set in `/configs/config.yml`).

### Visualizing a rollout

To visualize a rollout saved in `complex_envs/output.npz`, run the following command from the same folder:

```
# python visualize.py -i output.npz -f first_output
```

This will produce two files `first_output.eps` which plots the trajectory of the energy over the episode and `first_output.gif` which is an animation of the bridge and the impulse force.


