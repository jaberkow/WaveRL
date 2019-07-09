from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import gym
from gym.spaces import Box, Tuple, Discrete
from gym.envs.classic_control import rendering
from .finite_diff_wave import Wave1D
import random

import pickle
import os
import yaml

class VibratingBridge(gym.Env):
    """
    An environment that can be used to learn active damping control policies for
    a an oscillating bridge modeled with the one dimensional wave equation with
    endpoints fixed at zero.

    The goal of active damping is to apply external forces to a oscillating system
    to cancel out the oscillations and bring the system to rest.

    The forces are modeled as pistons at points along the interior of the bridge
    applying localized impulse forces.  Actions in this environment are equivalent
    to setting the forces of these pistons.
    """

    def __init__(self,config):
        """
        Constructor for the VibratingBridge OpenAI gym

        Inputs:
            config:  A dict containing parameters for the system, which must have the following keys:

            num_warmup_steps: (int > 0)
            num_equi_steps: (int > 0)
            num_force_points: (int > 0)
            min_force: (float) the maximum value of the applied forces
            max_force: (float) the minimum value of the applied forces
            min_u: (float) upper bound of system's deviation
            max_u: (float) lower bound of system's deviation
            timepoints_per_step: (int > 0) how many steps of the system dynamics to take per step of the
                environment
            max_steps:  maximum number of environments steps to take before ending episode
            num_lattice_points: (int > 0) how many discrete points along the length of the system to use for
                the finite difference scheme
            drive_magnitude: The L2 magnitude of the drivinge force of the warmup period


        """

        self.num_warmup_steps = config['num_warmup_steps']
        self.num_equi_steps = config['num_equi_steps']
        self.num_force_points = config['num_force_points']
        self.min_force = config['min_force']
        self.max_force = config['max_force']
        self.min_u = config['min_u']
        self.max_u = config['max_u']
        # How many steps of the dynamics to run in one step of the environment
        self.timepoints_per_step = config['timepoints_per_step']
        self.max_steps = config['max_steps']
        self.Nx = config['num_lattice_points']
        self.drive_magnitude = config['drive_magnitude']
        # Load the simulator class, thi is the line to change if you
        # use a different method for simulating the dynamics
        self.simulator = Wave1D(config)

        # Build up the action space
        self.action_space = Box(low=self.min_force,high=self.max_force,
                                shape=(self.num_force_points,),dtype=np.float32)
        # Build up the observation space
        self.observation_space = Box(low=self.min_u,high=self.max_u,
                                     shape=(1,self.Nx+1,3),dtype=np.float32)

        # Allocate for trajectories
        self.height_traj = []
        self.impulse_traj = []
        self.energy_traj = []
        self.code_traj = []
        self.reset()

    def reset(self):
        """
        This resets the state of the system.

        """
        # Reset the step_number
        self.step_number = 0
        # Clear out cache of trajectories
        self.height_traj = []
        self.impulse_traj = []
        self.energy_traj = []
        self.code_traj = [] # For rendering functions to track stage of simulation
        # Use the simulator's reset method, note that this also
        self.simulator.reset()

        # Random fixed action to warm up system
        action = self.action_space.sample()
        # Normalize it to make it larger
        action_mag = np.sqrt(np.sum(action**2))
        action *= self.drive_magnitude/action_mag
        self.simulator.take_in_action(action)
        # Run some warmup steps
        for t in range(self.num_warmup_steps):
            self.height_traj.append(np.copy(self.simulator.height))
            self.energy_traj.append(self.simulator.energy())
            self.impulse_traj.append(np.copy(self.simulator.get_impulse_profile()))
            self.simulator.single_step()
            self.code_traj.append(0)

        # Don't perturb system, let it equilibriate
        empty_action = 0.0*self.action_space.sample()
        self.simulator.take_in_action(empty_action)
        # equi_energy will be used for instance normalization
        self.equi_energy = 0
        for t in range(self.num_equi_steps):
            self.height_traj.append(np.copy(self.simulator.height))
            self.energy_traj.append(self.simulator.energy())
            self.equi_energy += self.simulator.energy()
            self.impulse_traj.append(np.copy(self.simulator.get_impulse_profile()))
            self.simulator.single_step()
            self.code_traj.append(1)
        # Divide equi_energy by num_equi_steps
        self.equi_energy /= self.num_equi_steps
        # Normalize the current energy_trajectory
        for i in range(len(self.energy_traj)):
            self.energy_traj[i] /= self.equi_energy


        observation = self.simulator.get_observation()
        return observation

    def step(self,action):
        """
        The main action step, we run a step of the finite difference scheme after
        we parameterize the impulse using parameters from action
        """

        # First we update the simulator's impulse profile using the action
        self.simulator.take_in_action(action)

        # Take in energy before running dynamics
        starting_energy = self.simulator.energy()/self.equi_energy

        # Run the dynamics with the fixed impulse for a fixed number of timepoints
        for t in range(self.timepoints_per_step):
            self.simulator.single_step()
            # Record things
            self.energy_traj.append(self.simulator.energy()/self.equi_energy)
            self.height_traj.append(np.copy(self.simulator.height))
            self.impulse_traj.append(np.copy(self.simulator.get_impulse_profile()))
            self.code_traj.append(2)

        # Take in energy after runing dynamics
        ending_energy = self.simulator.energy()/self.equi_energy

        # Reward is positive if energy is reduced
        reward = starting_energy - ending_energy

        observation = self.simulator.get_observation()
        # Properly bound the observation
        clipped_observation = np.clip(observation,self.min_u,self.max_u)

        # Update step number and check to see if epoch is over
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

        height_array = np.stack(self.height_traj,axis=1)
        impulse_array = np.stack(self.impulse_traj,axis=1)
        energy_array = np.array(self.energy_traj)
        code_array = np.array(self.code_traj,dtype=np.int32)
        np.savez(fname,height_array=height_array,impulse_array=impulse_array,
                 energy_array=energy_array,code_array=code_array,
                 x_mesh=self.simulator.x_mesh)
