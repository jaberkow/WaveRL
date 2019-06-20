from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import gym
from gym.spaces import Box, Tuple, Discrete
from gym.envs.classic_control import rendering
import random

import pickle
import os
import yaml

class ActiveDamping1D(gym.Env):
    """
    An environment that can be used to learn active damping control policies for
    a system modeled with the one dimensional wave equation with endpoints fixed
    at zero.

    Active damping is modeled as a gaussian impulse with scale factor alpha in 
    [-1,1] and center mu in [0,L] where L is the length of the system.  Alpha and
    mu are components of the agent actions.  The goal is to minize the L2 magnitude
    of the deviations 
    """

    def __init__(self,config):
        """
        Constructor for the ActiveDamping OpenAI gym

        Inputs:
            config:  A dict containing parameters for the system
        """
        self.Nt = config['num_time_points']
        self.dt = config['time_interval']
        self.c_speed = config['wave_speed']
        self.L = config['system_length']
        self.Nx = config['num_lattice_points']



        #starting profile is a superposition of waves
        self.Nwaves = config['num_waves'] 

        #the lattice spacing
        self.dx = float(self.L)/float(self.Nx)

        #the courant number
        self.C = self.c_speed *self.dt/self.dx
        self.C2 = self.C**2 #helper number

        #The effective time interval
        self.T= self.Nt*self.dt 
        #Mesh points in time
        self.t_mesh = np.linspace(0, self.T, self.Nt+1)
        #Mesh points in space
        self.x_mesh = np.linspace(0.0,self.L,self.Nx+1)

        #recalibrate the resolutions to account for rounding
        self.dx = self.x_mesh[1] - self.x_mesh[0]
        self.dt = self.t_mesh[1] - self.t_mesh[0]
        

        #The system is always initially at rest
        #TODO: generalize this by making V passable parameter
        self.V = lambda x: 0

        #allocate memory for the recursive solution arrays
        self.u     = np.zeros(self.Nx + 1)   # Solution array at new time level
        self.u_n   = np.zeros(self.Nx + 1)   # Solution at 1 time level back
        self.u_nm1 = np.zeros(self.Nx + 1)   # Solution at 2 time levels back

        #define the action space
        #the range of the scaling factor is configurable
        self.min_alpha = config['min_alpha']
        self.max_alpha = config['max_alpha']
        #the center of the impulse must lie in the system
        self.action_space = Box(low=np.array([self.min_alpha,0.0]),high=np.array([self.max_alpha,self.L]),dtype=np.float32)

        #the observation space will be the solution space essentially
        self.min_u = config['min_u']
        self.max_u = config['max_u']
        self.observation_space = Box(self.min_u, self.max_u, shape=(self.Nx + 1, ), dtype=np.float32)

        self.u_traj=[]
        self.action_traj=[]
        self.reset()

    def reset(self):
        """
        This resets the state of the system.

        """

        self.n = 0
        self.u_traj = []
        self.action_traj = []
        #First we set the impulse to zero
        self.impulse = lambda x:0.0

        #Now we create a functional defining the initial profile
        #It's a bunch of harmonics on L to preserve boundary conditions
        temp = np.zeros(self.Nx + 1)
        for i in range(self.Nwaves):
            wave_num = np.random.randint(low=-(self.Nx-1), high=self.Nx+1 )
            frequency = float(wave_num)*np.pi/float(self.L)
            temp += np.sin(self.x_mesh * frequency)
            
        #load in this initial condition to u_n
        self.u_n = np.copy(temp)

        #there's a special first step for the finite difference method
        for i in range(1, self.Nx):
            self.u[i] = self.u_n[i] + self.dt*self.V(self.x_mesh[i]) + 0.5*self.C2*(self.u_n[i-1] - 2*self.u_n[i] + self.u_n[i+1]) + 0.5*self.dt**2*self.impulse(self.x_mesh[i])
        #boundary conditions
        self.u[0] = 0;  self.u[self.Nx] = 0

        # Switch variables before next step
        self.u_nm1[:] = self.u_n
        self.u_n[:] = self.u
        return self.u

    def step(self,action):
        """
        The main action step, we run a step of the finite difference scheme after
        we parameterize the impulse using parameters from action
        """
        self.n+=1
        self.u_traj.append(self.u)
        self.action_traj.append(action)
        #setup our impulse function
        self.impulse = lambda x: action[0]*np.exp(-0.5*(x - action[1])**2)
        #the general step is different
        for i in range(1,self.Nx):
            self.u[i] = - self.u_nm1[i] + 2*self.u_n[i] + self.C2*(self.u_n[i-1] - 2*self.u_n[i] + self.u_n[i+1]) + self.dt**2*self.impulse(self.x_mesh[i])
        #boundary conditions
        self.u[0] = 0;  self.u[self.Nx] = 0

        # Switch variables before next step
        self.u_nm1[:] = self.u_n
        self.u_n[:] = self.u

        # we have to clip the variables now to lie in the box, hopefully this has no effect
        temp = np.clip(self.u, a_min=self.min_u, a_max=self.max_u)
        self.u = np.copy(temp)

        self.reward = -np.sum(self.u**2)

        #now we check if done
        done = (self.n >= self.Nt)
        return self.u,self.reward,done,{}

    def render(self,fname='testout'):
        """
        The render method just saves to file for later animation
        """
        u_array = np.concatenate(self.u_traj, axis=0)
        action_array = np.concatenate(self.action_traj,axis=0)
        np.savez(fname,u_array=u_array, action_array = action_array)









        


