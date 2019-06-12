#!/bin/bash

apt-get update
apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev openmpi-bin mpich lam-runtime
conda env create --file conda_requirements_baseline.yaml
source activate baseline_env
pip install stable-baselines