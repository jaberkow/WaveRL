#!/bin/bash

conda env create --file conda_requirements.yaml
source activate RL_control
pip install requests
pip install -U ray
pip install ray[rllib] 
