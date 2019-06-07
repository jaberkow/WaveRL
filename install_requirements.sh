#!/bin/bash

conda env create --file conda_requirements.yaml
source activate RL_control
pip install -U ray
pip install tensorflow
pip install ray[rllib] 
