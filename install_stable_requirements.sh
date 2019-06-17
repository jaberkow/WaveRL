#!/bin/bash

conda env create --file conda_requirements_baseline.yaml
source activate baseline_env
pip install stable-baselines
pip install pyyaml