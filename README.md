# WaveRL
A package for training RL agents to perform active damping on a model of a vibrating bridge.  [Link to project presentation](https://docs.google.com/presentation/d/11RFd_cj0aX3AAjNP0hrgOwF8einq7tsvyfX5sfljVik/edit?usp=sharing)

## Contents
* src/ :  This folder contains the code for the environments as well as scripts for training and rolling out agents
	* train.py :  This script trains an agent (see example below)
	* rollout.py : This script rolls out a trained agent (see example below)
	* visualize.py : This script produces visualizations of a rolled out agent (see example below)
	* environments/ : This folder contains code for the environments.
		* finite_diff_wave.py : This is a class definition for a simulator of one dimensional wave equation with finite difference methods.
		* active_damping_env.py : This is a class definition for an OpenAI gym environment simulating an oscillating bridge
* configs/
	* config.yml : This file holds the default parameters for the scripts and environments
* tests/
	* config_test.py :  A unnittest test fixture that can be used to make sure `configs/config.yml` has all the appropriate keys and valid parameter settings
* trained_agents/ : A folder for storing trained agents
* rollouts/ : A folder for storing rollouts of trained agents and associated visualizations.  Currently includes an example rollout and visualizations of a trained agent.
* install_stable_requirements.sh : a shell script for installing all the necessary packages
* conda_requirements_baseline.yaml : A specification of the conda environment

## Setup
First, [Install Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)

Then update and install the following system packages:

### Ubuntu Instructions:
```
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev openmpi-bin mpich lam-runtime
```
### Mac OS X Instructions:
Installing the necessary C-libraries is easiest with [Homebrew](https://brew.sh/), so install this first:
```
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```
Then install cmake and openmpi.
```
brew install cmake openmpi
```
### Setting up the conda environment

Clone the repository and change into the repo folder
```
git clone https://github.com/jaberkow/WaveRL.git
cd WaveRL
```

Make a conda environment and activate it:

```
conda create -n WaveEnv python=3.6
conda activate WaveEnv
```
Install the packages from `requirements.txt`

```
pip install -r requirements.txt
```
### Tests

If any changes are made to the default values in `configs/config.yml` , run the following command

```
python tests/config_test.py
```
To make sure that all the parameter values are valid.

## About the environment

This package simulates an oscillating bridge by modelling it with the one-dimensional [wave equation](https://en.wikipedia.org/wiki/Wave_equation), which is simulated using a [finite difference solver](https://en.wikipedia.org/wiki/Finite_difference_method).  The action space of the environment represents pistons that apply a force to actively dampen vibrations in the bridge.  The reward signal is proportional to the decrease in energy of the system.  One episode of the environment involves 3 phases:  1) A "warmup phase" where an external force is applied to the system to cause oscillations 2) An "equilibriation" phase where the oscillations settle in to stable patterns and 3) A dampening phase where the agent attempts to dampen the oscillations.

Here is an example of a single episode, the red line is the bridge and the green line reprents a smoothed profile of the forces applied to the bridge by the pistons.

![](rollouts/example_visualization.gif)

## Training an agent

To train an agent for 40,000 timesteps on the vibrating bridge environment and save it as `trained_agents/damping_agent.pkl`, run the following command:

```
python src/train.py -n 40000 -m trained_agents/damping_agent
```
Training may produce deprecationg warnings due to the version of TensorFlow used in the current release of the stable baselines package, which can be ignored.  This command will also produce a TensorBoard folder at `/tensorboard_log` that can be visualized by running the following command from the same directory and following the instructions in the terminal:

```
tensorboard --logdir tensorboard_log/
```
## Rolling out a trained agent

To rollout a trained agent that is stored at `trained_agents/damping_agent.pkl` for 60 steps, run the following command:

```
python src/rollout.py -n 60 -i trained_agents/damping_agent.pkl -f rollouts/damping_rollout.npz
```

The rollout will be saved as `rollouts/damping_rollout.npz`, which can be changed by passing `-f <filename>` to the above command.  Note that the trajectories produced will have length equal to the number of rollout steps + the number of warmup steps + the number of equilibriation steps (values of which are set in `/configs/config.yml`).

## Visualizing a rollout

To visualize a rollout saved in `rollouts/damping_rollout.npz`, run the following command:

```
python src/visualize.py -i rollouts/damping_rollout.npz -f rollouts/damping_visualiztion
```

This will produce two files `rollouts/damping_visualiztion.png` which plots the trajectory of the energy over the episode and `rollouts/damping_visualiztion.gif` which is an animation of the bridge and the impulse force.  The visualizations, png or gif, can be viewed from the command line in linux with

```
animate <path_to_file>
```
and in MacOS with

```
qlmanage -p <path_to_file>
```

## Exploring parameter values

The parameters that govern the vibrating bridge environment (as well as default parameters for training and rollout) are set in `configs/config.yml`.  There are several parameters that may be interesting to alter:

* wave_speed :  This value controls how fast a wave propagates along the bridge.  Larger values yield a more 'taut' bridge and smaller values yield a 'looser' bridge.  Must be strictly greater than zero.
* force_width :  Currently the piston forces are modeled as having Gaussian profiles centered at discrete points with widths given by this parameter.  Decreasing this value will the make the forces more point-like.  Must be strictly greater than zero.
* num_force_points :  The number of pistons.  Increasing this parameter while decreasing the force_width model's an active damping system capable of more fine grained control.  Must be a positive int.
* timepoints_per_step : How many steps of the simulator dynamics to run with a fixed value of the piston forces.  Increasing this parameter decreases the power of the agent/damping system to respond quickly.  Must be a positive int.

### Rolling out an agent trained on an environment with different parameters

If you are interested in judging how well an agent trained with one set of parameters governing the vibrating bridge environment generalizes to an environment with different parameters you can train an agent, then change **some** parameters (see below) in the configuration file (`/configs/config.yml`), and roll out the `.pkl` file of the trained agent in the normal way.  However, several parameters **must** remain constant between training and rolling out, or else the OpenAI gym will throw an error because the observation/action spaces have changed.  These fixed parameters are as follows:

* num_lattice_points
* num_force_points
* min_force
* max_force
* min_u
* max_u
