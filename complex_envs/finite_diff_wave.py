import numpy as np
from scipy.integrate import simps
class Wave1D:
    """
    A utility class for simulating the wave equation in 1 dimension
    """
    def __init__(self,config):
        """
        Constructor 1 dimensional wave system

        Inputs:
            config:  A dict containing parameters for the system
        """
        
        self.dt = config['time_interval']
        self.c_speed = config['wave_speed']
        self.L = config['system_length']
        self.Nx = config['num_lattice_points']
        #how many points along the domain can impulse force be applied
        self.num_force_points = config['num_force_points']
        #set the locations of the force application
        self.force_locations = np.linspace(0.0,self.L,self.num_force_points+2)[1:self.num_force_points+1]
        #how wide is the profile of each impulse force, must be > 0
        self.force_width = config['force_width']
        #scale the force width by system length
        self.force_width *= self.L

        #the lattice spacing
        self.dx = float(self.L)/float(self.Nx)

        #Mesh points in space
        self.x_mesh = np.linspace(0.0,self.L,self.Nx+1)
        
        #the courant number
        self.C = self.c_speed *self.dt/self.dx
        self.C2 = self.C**2 #helper number

        #recalibrate the resolutions to account for rounding
        self.dx = self.x_mesh[1] - self.x_mesh[0]
        
        #We set up the conditions of the system before warmup period

        #The system is always initially at rest
        self.V = lambda x: 0
        
        #we assume the system starts completely flat
        #TODO: generalize this
        self.I = lambda x: 0
        

        #allocate memory for the recursive solution arrays
        self.u     = np.zeros(self.Nx + 1)   # Solution array at new time level
        self.u_n   = np.zeros(self.Nx + 1)   # Solution at 1 time level back
        self.u_nm1 = np.zeros(self.Nx + 1)   # Solution at 2 time levels back


        self.u_traj=[]
        self.action_traj=[]
        self.reset()
    
    def reset(self):
        """
        Resets the state of the wave system
        """
        #we reset the time and step index
        self.t = 0
        self.n = 0
        
        #we set the force vals to zero
        self.force_vals = np.zeros(self.num_force_points)
        
        
        #we set the initial condition of the solution 1 time level back
        for i in range(0,self.Nx+1):
            self.u_n[i]=self.I(self.x_mesh[i])
        
        #We do a special first step for the finite difference scheme
        #note that the source term is just the driving force for now
        for i in range(1,self.Nx):
            self.u[i] =self.u_n[i] + self.dt*self.V(self.x_mesh[i])
            self.u[i]+=0.5*self.C2*(self.u_n[i-1] - 2*self.u_n[i] + self.u_n[i+1])
            self.u[i]+=0.5*(self.dt**2)*self.impulse_term(self.x_mesh[i])
        #force boundary conditions
        self.u[0]=0
        self.u[self.Nx]=0
        #switch solution steps
        self.u_nm1[:] = self.u_n
        self.u_n[:] = self.u
    
    def single_step(self):
        """
        Run a single step of the wave equation finite difference dynamics
        """
        
        self.t += self.dt
        self.n += 1
        for i in range(1,self.Nx):
            self.u[i] = -self.u_nm1[i] + 2*self.u_n[i] 
            self.u[i] += self.C2*(self.u_n[i-1] - 2*self.u_n[i] + self.u_n[i+1]) 
            self.u[i] += (self.dt**2)*self.impulse_term(self.x_mesh[i])
        #force boundary conditions
        self.u[0] = 0  
        self.u[self.Nx] = 0

        #switch solution steps
        self.u_nm1[:] = self.u_n
        self.u_n[:] = self.u
    
    def take_in_action(self,action):
        """
        This method acts as the interface where the agent applies an action to environment.
        For this simulator, it's simply a setter method for the force_vals attribute that
        determine the profile of the impulse term.
        """
        self.force_vals = np.copy(action)
    
    def impulse_term(self,x):
        """
        The function definition for the active damping terms
        
        Inputs:
            x - a scalar, position in the domain
            force_vals - A vector of shape (self.num_force_points),
                the (signed) values of the force at each piston point
        """
        return np.sum(self.force_vals*np.exp(-0.5* ((x-self.force_locations)**2 )/self.force_width))
    
    def get_impulse_profile(self):
        """
        A utility function for returning an array representing the shape of the resulting impulse 
        force, this is used for rendering the history of actions taken by the agent.
        
        Inputs:
            force_vals - A vector of shape (self.num_force_points),
                the (signed) values of the force at each piston point
        """
        profile = []
        for i in range(self.Nx+1):
            profile.append(self.impulse_term(self.x_mesh[i]))
        return np.array(profile)
    
    def get_observation(self):
        """
        This is an interface that returns the observation of the system, which is modeled
        as the state of the wave system for the current timestep, previous timestep, and 
        twice previous timestep.
        
        Outputs:
            observation - An array of shape (1,self.Nx+1,3).  observation[0,:,0]= self.u,
                observation[0,:,1]=self.u_n, and observation[0,:,2]=self.u_nm1 
        """
        
        observation = np.zeros((1,self.Nx+1,3))
        observation[0,:,0]= self.u
        observation[0,:,1]=self.u_n
        observation[0,:,2]=self.u_nm1
        return observation

    def energy(self):
        """
        Computes the internal energy of the system based upon the integral functional for
        the 1-D wave equation.

        There will be inherent periodicity even in the absence of external driving force,
        due to the finite difference approximation.

        See http://web.math.ucsb.edu/~grigoryan/124A/lecs/lec7.pdf for details
        """

        dudt = (self.u-self.u_nm1)/self.dt #time derivative
        dudx = np.gradient(self.u,self.x_mesh) #space derivative
        
        space_term = -self.u*np.gradient(dudx,self.x_mesh) #alternative tension energy
        energy_density = dudt**2 + (self.C**2)*(dudx**2)
        #energy_density = dudt**2 + (self.c_speed**2)*space_term
        return 0.5*simps(energy_density,self.x_mesh)