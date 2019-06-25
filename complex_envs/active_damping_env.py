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
        
        self.num_warmup_steps = config['num_warmup_steps']
        self.num_equi_steps = config['num_equi_steps']
        self.num_force_points = config['num_force_points']
        self.min_force = config['min_force']
        self.max_force = config['max_force']
        self.min_u = config['min_u']
        self.max_u = config['max_u']
        #how many steps of the dynamics to run in one step of the environment
        self.timepoints_per_step = config['timepoints_per_step']
        self.max_steps = config['max_steps']
        self.Nx = config['num_lattice_points']
        
        self.simulator = fdw.Wave1D(config)
        
        #build up the action space
        self.action_space = Box(low=self.min_force,high=self.max_force,
                                shape=(self.num_force_points,),dtype=np.float32)
        #build up the observation space
        self.observation_space = Box(low=self.min_u,high=self.max_u,
                                     shape=(1,self.Nx+1,3),dtype=np.float32)
        
        #allocate for trajectories
        self.u_traj = []
        self.impulse_traj = []
        self.energy_traj = []
        self.code_traj = []
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
        self.energy_traj = []
        self.code_traj = [] #for rendering functions to track stage of simulation
        #use the simulator's reset method, note that this also
        self.simulator.reset()
        
        #run some warmup steps
        for t in range(self.num_warmup_steps):
            #random actions to warm up system
            action = self.action_space.sample()
            self.u_traj.append(np.copy(self.simulator.u))
            self.energy_traj.append(self.simulator.energy())
            self.simulator.take_in_action(action)
            self.impulse_traj.append(np.copy(self.simulator.get_impulse_profile()))
            self.simulator.single_step()
            self.code_traj.append(0)
        
        #don't perturb system, let it equilibriate
        empty_action = 0.0*self.action_space.sample()
        self.simulator.take_in_action(empty_action)
        for t in range(self.num_equi_steps):
            self.u_traj.append(np.copy(self.simulator.u))
            self.energy_traj.append(self.simulator.energy())
            self.impulse_traj.append(np.copy(self.simulator.get_impulse_profile()))
            self.simulator.single_step()
            self.code_traj.append(1)
        

        observation = self.simulator.get_observation()
        return observation

    def step(self,action):
        """
        The main action step, we run a step of the finite difference scheme after
        we parameterize the impulse using parameters from action
        """
        
        #first we update the simulator's impulse profile using the action
        self.simulator.take_in_action(action)
        
        #take in energy before running dynamics
        starting_energy = self.simulator.energy()
        
        #run the dynamics with the fixed impulse for a fixed number of timepoints
        for t in range(self.timepoints_per_step):
            self.simulator.single_step()
            #record things
            self.energy_traj.append(self.simulator.energy())
            self.u_traj.append(np.copy(self.simulator.u))
            self.impulse_traj.append(np.copy(self.simulator.get_impulse_profile()))
            self.code_traj.append(2)
        
        #take in energy after runing dynamics
        ending_energy = self.simulator.energy()
        
        #reward is positive if energy is reduced
        reward = starting_energy - ending_energy
        
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
        energy_array = np.array(self.energy_traj)
        code_array = np.array(self.code_traj,dtype=np.int32)
        np.savez(fname,u_array=u_array,impulse_array=impulse_array,
                 energy_array=energy_array,code_array=code_array,
                 x_mesh=self.simulator.x_mesh)









        


