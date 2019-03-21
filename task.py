import numpy as np
from physics_sim import PhysicsSim
import math

sigmoid = lambda x: 1 / (1 + math.exp(-x))


class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""

    def __init__(self, init_pose=None, init_velocities=None,
                 init_angle_velocities=None, runtime=5., target_pos=None, target_height=5.):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 1

        self.reached_target_height = False
        self.target_height = target_height
        self.state_size = self.action_repeat * 6
        self.action_low = 402
        self.action_high = 440
        self.action_size = 4
        self.current_speeds = [0, 0, 0, 0]
        self.runtime = runtime

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def get_reward(self):
        #target_pos_reward = np.tanh(1. - .01 * (abs(self.sim.pose[:3] - self.target_pos)).sum())
        height_reward = (self.target_height - self.sim.pose[2]) / self.target_height
        prop_speed_difference = - np.std(self.current_speeds) / 10
        flight_time = self.sim.runtime / self.runtime
        #print ("target_pos_reward: " + str(target_pos_reward))
        #print ("prop_speed_difference: " + str(prop_speed_difference))
        #print ("flight_time: " + str(flight_time))

        reward = 0.4 * prop_speed_difference + \
               0.3 * flight_time + \
               0.3 * height_reward
        #print ("Reward: " + str(reward))
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        self.current_speeds = rotor_speeds
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds)  # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        self.reached_target_height = False
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state
