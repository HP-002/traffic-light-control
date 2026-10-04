from abc import ABC, abstractmethod

import numpy as np

from environment.traffic_light import TrafficLightEnv

class Agent(ABC):
    def __init__(
        self,
        env: TrafficLightEnv,
        learning_rate=0.1,
        discount_factor=0.95,
        initial_epsilon=1.0,
        epsilon_decay=0.95,
        minimum_epsilon=0.05,
        seed=42,
    ):
        # Environment
        self.env = env

        # Learning Rate or alpha
        self.lr = learning_rate
        # Discount Factor or gamma
        self.discount_factor = discount_factor

        # Epsilon - Used for epsilon-greedy algorithm
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = minimum_epsilon

        # Seed for randomness
        self.seed = seed

        # Observation space is [Light, N traffic, S traffic, E traffic, W traffic].
        # The dimensions of each are [9, 4, 4, 4, 4]
        self.state_dimensions = tuple(int(n) for n in env.observation_space.nvec) # (9, 4, 4, 4, 4)
        self.n_states = np.prod(self.state_dimensions)
        self.n_actions = env.action_space.n

        # Log
        self.reward_per_episode = []
        self.epsilon_per_episode = []

        self._init_tables()

    @abstractmethod
    def _init_tables(self):
        """Create and initialize Q tables with zeros."""

    @abstractmethod
    def get_q_tables(self):
        """Returns the Q table. If multiple, returns as a tuple."""

    @abstractmethod
    def update_q_value(self, state, action, reward, next_state):
        """Update the value in the Q-table based on the agent's algorithm."""

    def update_epsilon(self):
        """Update epsilon value based on decay."""
        self.epsilon = max(self.min_epsilon, self.epsilon_decay * self.epsilon)

    def get_state_row(self, state):
        """Get the row index based on the state"""
        s = [int(x) for x in state]
        return np.ravel_multi_index(s, self.state_dimensions)
