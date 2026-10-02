import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces

from states import LightState, TrafficState
from actions import Actions

class TrafficLightEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        max_steps=25,
        max_cars_to_proccess=15,
        max_traffic_count=5,
        stochastic=False,
        seed=42,
    ):
        """
        Initialize the class
        """
        # Maximum steps to take by agent
        self.max_steps = max_steps

        # maximum cars to process
        self.max_cars_to_proccess = max_cars_to_proccess

        # maximum cars present in a single side at any given time
        self.max_traffic_count = max_traffic_count

        # If true, cars arrive in a random way
        self.stochastic = stochastic
        self.seed = seed

        # Action space
        self.action_space = spaces.Discrete(len(Actions))
        # Observation space
        # Each State is [Light State, North Traffic, South Traffic, East Traffic, West Traffic]
        self.observation_space = spaces.MultiDiscrete([len(LightState)] + [self.max_traffic_count + 1] * 4)

        self.step_count = 0
        self.cars_processed = 0

        # Initial State
        self.light_state = LightState.ALL_RED
        self.traffic_count = np.array([0] * 4) # N, S, E, W

    def _get_obs(self):
        """Convert internal state to observation format

        Returns:
            np.array: Array with [Light State, N traffic, S traffic, E traffic, W traffic]
        """
        return np.array([
            self.light_state, 
            self.traffic_count[0], 
            self.traffic_count[1], 
            self.traffic_count[2], 
            self.traffic_count[3]
        ], dtype=np.int64)

    def _get_info(self):
        """Format internal information about parameters
        
        Returns:
            dict: containing all internal parameters and information
        """
        info = {
                    "max_step_count": self.max_steps,
                    "max_cars_to_proccess": self.max_cars_to_proccess,
                    "max_traffic_count": self.max_traffic_count,
                    "stochastic": self.stochastic,
                    "seed": self.seed,
                    "step_count": self.step_count,
                    "cars_processed": self.cars_processed,
                    "light_state": self.light_state,
                    "traffic_count": self.traffic_count,
                }
        return info

    def reset(self):
        """
        Reset the state of the environment to an initial state
        """
        super().reset(seed=self.seed)

        self.step_count = 0
        self.cars_processed = 0

        self.light_state = LightState.ALL_RED
        self.traffic_count = np.array([0] * 4) # N, S, E, W

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def _allowed_light_change(self, next_state):
        allowed_changes = {
            LightState.ALL_RED: {
                LightState.ALL_RED,
                LightState.NS_GREEN,
                LightState.EW_GREEN,
            },
            LightState.NS_GREEN: {
                LightState.NS_GREEN,
                LightState.NS_YELLOW,
            },
            LightState.NS_YELLOW: {
                LightState.ALL_RED,
            },
            LightState.EW_GREEN: {
                LightState.EW_GREEN,
                LightState.EW_YELLOW
            },
            LightState.EW_YELLOW: {
                LightState.ALL_RED,
            }
        }

        return next_state in allowed_changes[self.light_state]

    def _add_car(self):
        """Add a car to the intersection"""
        if self.stochastic:
            pass
        else:
            self

    def _process_car(self):
        """Remove a car from the intersection if possible"""
        pass

    def _get_reward(self):
        """Calculate the reward based on the number of cars processed"""
        pass
    
    def step(self, action: Actions):
        """
        Execute one timestep within the environment

        Args:
            action: Action to take
        """
        pass

    def render(self):
        """
        Visualize the environment
        """
        pass