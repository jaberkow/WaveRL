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

class SimpleLineViewer:
    """
    This class is for creating simple plots of a one dimensional time series
    """
    def __init__(self):
        
        pass
    def render(self,trajectory,fname='output'):
        """
        This method is the actual render operation of the simple corridor class

        Inputs:
            trajectory:  Array of shape (num_timesteps,), the agent's trajectory over time
            save: Boolean, whether or not to save the trajectory to a file
        """
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.plot(range(len(trajectory)),trajectory)
        fig.savefig(fname)
    def close(self):
        plt.close()



class SimpleCorridor(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.
    You can configure the length of the corridor via the env config."""

    def __init__(self, config):
        """
        Constructor for the simple corridor OpenAI gym

        Inputs:
            config:  A dict, must contain a 'corridor_length' entry
        """
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

    def render(self,fname='output',mode='human'):
        """
        Saves a png file showing the trajectory of the agent over time
        """
        if self.viewer == None:
            #Build the viewing object
            self.viewer=SimpleLineViewer()
        self.viewer.render(self.trajectory,fname=fname)
    def close(self):
        if self.viewer != None:
            self.viewer.close()
            self.viewer = None