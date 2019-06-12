# ComplexRL
A package for training RL agents in complex physical environments



## Setup
First, [Install Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)

Then update and install the following system packages:

### Ubuntu Instructions:
```
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev openmpi-bin mpich lam-runtime
```
### Mac OS X Instructions:
```
brew install cmake openmpi
```
### Setting up the conda environment

Clone the repository
```
$ git clone https://github.com/jaberkow/Insight_Project.git
```
Make sure 'install_requirements.sh' has the correct permissions and run it.

```
$ chmod 755 install_stable_requirements.sh
$ ./install_stable_requirements.sh
```
Activate the newly created environment

```
$ conda activate baseline_env
```
