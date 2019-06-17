from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gym
from gym.spaces import Box, Tuple, Discrete
from gym.envs.classic_control import rendering
import random

import pickle
import os
import yaml

from ray.rllib.models import FullyConnectedNetwork, Model, ModelCatalog

import ray
from ray import tune
from ray.tune import grid_search
from ray.tune.registry import register_env

class SimpleLineViewer:
    def __init__(self):
        fig = plt.figure()
        plt.show(block=False)
    def render(self,trajectory):
        plt.plot(range(len(trajectory)),trajectory)
        plt.pause(0.001)



class SimpleCorridor(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.
    You can configure the length of the corridor via the env config."""

    def __init__(self, config):
        self.end_pos = config["corridor_length"]
        self.cur_pos = 0
        self.action_space = Discrete(2)
        self.observation_space = Box(
            0.0, self.end_pos, shape=(1, ), dtype=np.float32)

        # For rendering:
        self.viewer = None
        self.trajectory=[]

    def reset(self):
        self.cur_pos = 0
        self.trajectory=[]
        return [self.cur_pos]

    def step(self, action):
        assert action in [0, 1], action
        if action == 0 and self.cur_pos > 0:
            self.cur_pos -= 1
        elif action == 1:
            self.cur_pos += 1
        done = self.cur_pos >= self.end_pos
        self.trajectory.append(self.cur_pos)
        return [self.cur_pos], 1 if done else 0, done, {}

    def render(self,mode='human'):
        if self.viewer == None:
            #Build the viewing object
            self.viewer=SimpleLineViewer()
            self.viewer.render(self.trajectory)


class CustomModel(Model):
    """Example of a custom model.
    This model just delegates to the built-in fcnet.
    """

    def _build_layers_v2(self, input_dict, num_outputs, options):
        self.obs_in = input_dict["obs"]
        self.fcnet = FullyConnectedNetwork(input_dict, self.obs_space,
                                           self.action_space, num_outputs,
                                           options)
        return self.fcnet.outputs, self.fcnet.last_layer

if __name__ == "__main__":
    # Can also register the env creator function explicitly with:
    register_env("corridor", lambda config: SimpleCorridor(config))
    ray.init()
    ModelCatalog.register_custom_model("my_model", CustomModel)
    tune.run(
        "PPO",
        stop={
            "timesteps_total": 10000,
        },
        config={
            "env": SimpleCorridor,  # or "corridor" if registered above
            "model": {
                "custom_model": "my_model",
            },
            "lr": grid_search([1e-2, 1e-4, 1e-6]),  # try different lrs
            "num_workers": 1,  # parallelism
            "env_config": {
                "corridor_length": 5,
            },
        },
        checkpoint_freq=50,
        checkpoint_at_end=True
    )