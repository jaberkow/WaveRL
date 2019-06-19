import numpy as np

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

        #driving force profile is a superposition of waves
        self.Nwaves = config['num_waves'] 
        self.drive_period = config['drive_period']
        self.drive_frequency = 2.0*np.pi/self.drive_period

        #the lattice spacing
        self.dx = float(self.L)/float(self.Nx)

        #the courant number
        self.C = self.c_speed *self.dt/self.dx
        self.C2 = self.C**2 #helper number

        #Mesh points in space
        self.x_mesh = np.linspace(0.0,self.L,self.Nx+1)

        #recalibrate the resolutions to account for rounding
        self.dx = self.x_mesh[1] - self.x_mesh[0]
        

        #The system is always initially at rest
        #TODO: generalize this by making V passable parameter
        self.V = lambda x: 0
        
        #we assume the system starts completely flat
        self.I = lambda x: 0
        
        #setup the source term
        self.source_term = lambda x,t: 0

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
        #We reset the driving force
        self.setup_driving_force()
        #set up a source term that is just the driving force for now
        self.source_term = lambda x,t : self.driving_force(x,t)
        #we set the initial condition of the solution 1 time level back
        for i in range(0,self.Nx+1):
            self.u_n[i]=self.I(self.x_mesh[i])
        
        #We do a special first step for the finite difference scheme
        #note that the source term is just the driving force for now
        for i in range(1,self.Nx):
            self.u[i] =self.u_n[i] + self.dt*self.V(self.x_mesh[i])
            self.u[i]+=0.5*self.C2*(self.u_n[i-1] - 2*self.u_n[i] + self.u_n[i+1])
            self.u[i]+=0.5*(self.dt**2)*self.source_term(self.x_mesh[i],self.t)
        #force boundary conditions
        self.u[0]=0
        self.u[self.Nx]=0
        #switch solution steps
        self.u_nm1[:] = self.u_n
        self.u_n[:] = self.u
    
    def setup_driving_force(self):
        """
        Creates the external driving force that is oscillating the bridge
        
        It has a spatial profile that is a sum of harmonics of the system,
        and an oscillating temporal profile.
        """
        frequency_list = []
        for i in range(self.Nwaves):
            wave_num = np.random.randint(low=-(self.Nx-1), high=self.Nx+1 )
            frequency = float(wave_num)*np.pi/float(self.L)
            frequency_list.append(frequency)
        self.spatial_freq_array = np.array(frequency_list)
        
    def driving_force(self,x,t):
        """
        The driving force function.
        
        Inputs:
            x - a scalar, position in the domain
            t - a scalar, current timestep
        """
        return np.sin(self.drive_frequency*t)*np.sum(np.sin(x*self.spatial_freq_array))
    
    def single_step(self):
        """
        Run a single step of the wave equation finite difference dynamics
        """
        
        self.t += self.dt
        self.n += 1
        for i in range(1,self.Nx):
            self.u[i] = -self.u_nm1[i] + 2*self.u_n[i] 
            self.u[i] += self.C2*(self.u_n[i-1] - 2*self.u_n[i] + self.u_n[i+1]) 
            self.u[i] += self.dt**2*self.source_term(self.x_mesh[i],self.t)
        #force boundary conditions
        self.u[0] = 0  
        self.u[self.Nx] = 0

        #switch solution steps
        self.u_nm1[:] = self.u_n
        self.u_n[:] = self.u
    def impulse_term(self,x,force_vals):
        """
        The function definition for the active damping terms
        
        Inputs:
            x - a scalar, position in the domain
            force_vals - A vector of shape (self.num_force_points),
                the (signed) values of the force at each piston point
        """
        return np.sum(force_vals*np.exp(-0.5* ((x-self.force_locations)**2 )/self.force_width))
    def take_in_action(self,action):
        """
        This function takes in the action from the agent and translates it into a change
        into the wave system's source term.
        
        
        """
        self.source_term = lambda x,t : self.driving_force(x,t) + self.impulse_term(x,action)
    def get_impulse_profile(self,force_vals):
        """
        A utility function for returning an array representing the shape of the resulting impulse force
        
        Inputs:
            force_vals - A vector of shape (self.num_force_points),
                the (signed) values of the force at each piston point
        """
        profile = []
        for i in range(self.Nx+1):
            profile.append(self.impulse_term(self.x_mesh[i],force_vals))
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