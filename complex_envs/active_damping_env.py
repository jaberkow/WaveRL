from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import gym
from gym.spaces import Box, Tuple, Discrete
from gym.envs.classic_control import rendering
import finite_diff_wave as fdw
import random

import pickle
import os
import yaml

class VibratingBridge(gym.Env):
    """
    An environment that can be used to learn active damping control policies for
    a an oscillating bridge modeled with the one dimensional wave equation with 
    endpoints fixed at zero.

    Active damping is modeled as pistons at points along the interior of the bridge
    applying localized impulse forces
    """

    def __init__(self,config):
        """
        Constructor for the VibratingBridge OpenAI gym

        Inputs:
            config:  A dict containing parameters for the system
        """
        
        self.num_warmup_step = config['num_warmup_steps']
        self.num_force_points = config['num_force_points']
        self.min_alpha = config['min_alpha']
        self.max_alpha = config['max_alpha']
        self.min_u = config['min_u']
        self.max_u = config['max_u']
        #how many steps of the dynamics to run in one step of the environment
        self.timepoints_per_step = config['timepoints_per_step']
        self.max_steps = config['max_steps']
        self.Nx = config['num_lattice_points']
        
        self.simulator = fdw.Wave1D(config)
        
        #build up the action space
        self.action_space = Box(low=self.min_alpha,high=self.max_alpha,
                                shape=(self.num_force_points,),dtype=np.float32)
        #build up the observation space
        self.observation_space = Box(low=self.min_u,high=self.max_u,
                                     shape=(1,self.Nx+1,3),dtype=np.float32)
        
        #allocate for trajectories
        self.u_traj = []
        self.impulse_traj = []
        self.reset()

    def reset(self):
        """
        This resets the state of the system.

        """
        #reset the step_number
        self.step_number = 0
        #clear out cache of trajectories
        self.u_traj = []
        self.impulse_traj = []
        #use the simulator's reset method, note that this also
        #resets the driving force
        self.simulator.reset()
        
        #run some warmup steps
        for t in range(self.num_warmup_step):
            self.simulator.single_step()

        observation = self.simulator.get_observation()
        return observation

    def step(self,action):
        """
        The main action step, we run a step of the finite difference scheme after
        we parameterize the impulse using parameters from action
        """
        
        #first we update the simulator's impulse profile using the action
        self.simulator.take_in_action(action)
        
        reward = 0
        
        #run the dynamics with the fixed impulse for a fixed number of timepoints
        for t in range(self.timepoints_per_step):
            self.simulator.single_step()
            self.u_traj.append(np.copy(self.simulator.u))
            self.impulse_traj.append(np.copy(self.simulator.get_impulse_profile(action)))
            #accumulate a reward
            reward -= self.simulator.get_loss()
            
            
        
        observation = self.simulator.get_observation()
        #properly bound the observation
        clipped_observation = np.clip(observation,self.min_u,self.max_u)
        
        #update step number and check to see if epoch is over
        self.step_number += 1
        if self.step_number >= self.max_steps:
            done = True
        else:
            done = False
        
        return clipped_observation,reward,done,{}

    def render(self,fname='testout'):
        """
        The render method just saves to file for later animation
        """
        
        u_array = np.stack(self.u_traj,axis=1)
        impulse_array = np.stack(self.impulse_traj,axis=1)
        np.savez(fname,u_array=u_array,impulse_array=impulse_array,x_mesh=self.simulator.x_mesh)









        


