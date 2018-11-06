import numpy as np
from tasks.physics_sim import PhysicsSim


class Task():
    """Task (environment) that defines the goal and provides feedback to the agent.
       The task is to reach a target pose.
    """
    
    def __init__(self, init_pose=None, init_velocities=None, init_angle_velocities=None, runtime=4., target_pos=None):
        
        """Initialize a Task object.
        
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
            target_v: target/goal (x,y,z) velocities for the agent
            target_angular_v: target/goal (x,y,z) angular velocities for the agent
        """
        
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([10.,10., 0.])
        #self.target_v = target_v if target_v is not None else np.array([0., 0., 0.])  
        #self.target_angular_v = target_angular_v if target_angular_v is not None else np.array([0., 0., 0.])  

    def get_reward(self):
        
        """Uses current pose of sim to return reward."""
        
        # the distance to the goal
        dist = np.linalg.norm(self.sim.pose[:3] - self.target_pos)
        abs_dist = abs(self.sim.pose[:3] - self.target_pos).sum()
        
        # the speed
        #speed = np.linalg.norm(self.sim.v)
        #abs_vel = abs(self.sim.v).sum()
        
        # the angular speed
        #ang_speed = np.linalg.norm(self.sim.angular_v)
        #abs_ang_vel = abs(self.sim.angular_v).sum()
        
        rew_fct =  np.tanh(1 - 0.0008 * abs_dist) # - 0.0001 * abs_vel)
        reward=rew_fct
        return reward
        
        
        ## sample reward function:
        #reward = 1.-.003*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        
        # the reward function - using just discounts and powers
        #rew_fct = 1 - (dist/600)**0.4
        
        # the reward function - using exponentials 
        #dist_reward = np.exp(- .001 * abs_dist)
        #vel_reward =  np.exp(- .01 * abs_vel)
        #ang_vel_reward = np.exp(- .01 * abs_ang_vel)
        #rew_fct = dist_reward * vel_reward * ang_vel_reward
        
        # the reward function - using arctan
        #rew_fct = np.arctan(1/dist**2)+0.01*np.arctan(1/speed**2)+0.01*np.arctan(1/ang_speed**2)    
        #rew_fct = np.arctan(1/abs_dist)+0.01*np.arctan(1/abs_vel)+0.001*np.arctan(1/abs_ang_vel)    
        
        # the reward function - using tanh
        #rew_fct =  np.tanh(1 - 0.0001 * dist - 0.0002 * vel - 0.0002 * ang_vel)
           
    
      
    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
            next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state