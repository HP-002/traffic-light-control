from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from environment.traffic_light import TrafficLightEnv
from environment.states import LightState, TrafficState, ExtraTimeState
from environment.actions import Actions

class TabularAgent(ABC):
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
        self.rng = np.random.default_rng(np.random.SeedSequence(self.seed).spawn(1)[0])

        # Observation space is [Light, N traffic, S traffic, E traffic, W traffic, Extra time].
        # The dimensions of each are [9, 4, 4, 4, 4, 2]
        self.state_dimensions = tuple(int(n) for n in env.observation_space.nvec) # (9, 4, 4, 4, 4, 2)
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
        """Returns 2 Q tables along with a boolean flag.
        - If the flag is True, 2 tables exist (Double Q-Learning) and both are returned.
        - If the flag is False, 1 table exists (Q-Learning) and the second value returned will be None.
        """

    @abstractmethod
    def get_action_values(self, s):
        """Returns the values of the actions for the given state from the Q table(s)"""

    @abstractmethod
    def update_q_value(self, state, action, reward, next_state, terminated):
        """Update the value in the Q-table based on the agent's algorithm."""

    def update_epsilon(self):
        """Update epsilon value based on decay.
        New epsilon value is `max(min_epsilon, epsilon_decay * epsilon)`
        """
        self.epsilon = max(self.min_epsilon, self.epsilon_decay * self.epsilon)

    def get_state_row(self, state):
        """Get the row index based on the state (Light, N traffic, S traffic, E traffic, W traffic).

        Args:
            state: The state in the format (Light, N, S, E, W) as returned by `_get_obs()` by the environment.
        Returns:
            `int`: the row index of the state `state` in the Q Table.
        """
        s = [int(x) for x in state]
        return np.ravel_multi_index(s, self.state_dimensions)

    def get_reward_per_episode(self):
        """Returns total reward per episode.
        
        Returns:
            `list`: of size `n_episodes` with total reward for that episode.
        """
        return self.reward_per_episode

    def get_epsilon_per_episode(self):
        """Returns epsilon values per episode.
        
        Returns:
            `list`: of size `n_episodes` with epsilon value for that episode.
        """
        return self.epsilon_per_episode

    def print_q_table(self, filename=None):
        """Print the Q-table as a labelled pandas dataframe and saves it to `filename` if provided.

        For Double Q-Learning the two tables are averaged.

        Returns:
            `pd.DataFrame`: the Q-table indexed by state, with actions as columns.
        """
        Q1, Q2, flag = self.get_q_tables()

        Q = (Q1 + Q2) / 2 if flag else Q1

        # Enum per observation dimension, matching the env's observation space layout
        state_enums = [LightState] + [TrafficState] * self.env.NUM_SIDES + [ExtraTimeState]
        states = [
            "(" + ", ".join(enum(v).name for enum, v in zip(state_enums, s, strict=True)) + ")"
            for s in np.ndindex(self.state_dimensions)
        ]
        actions = [Actions(a).name for a in range(self.n_actions)]

        df = pd.DataFrame(Q, index=states, columns=actions)

        print("Q Table:")
        print(df)
        
        if filename:
            df.to_csv(filename)

        return df

    def greedy_action(self, s):
        """Choose the greedy action for the given state s
        
        Returns
            int: The best action to take for the given state `s`
        """
        return self.random_argmax(self.get_action_values(s))

    def random_argmax(self, values):
        """Index of the maximum value in `values`, breaking ties uniformly at random.

        Returns:
            int: Index of a maximum value
        """
        return int(self.rng.choice(np.flatnonzero(values == values.max())))

    def choose_action(self, s):
        """Epsilon-Greedy policy.
        - Chooses random action with a probability of epsilon
        - Chooses greedy action with a probability of 1 - epsilon

        Returns:
            int: An action to take for the given state `s` 
        """
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.greedy_action(s)
    
    def train(self, n_episodes=10):
        """Train the agent for given number of episodes.
        Repeated calls continue training and append to the reward and epsilon logs.

        Args:
            `n_episodes`: The number of episodes to train for. default = 10
        """
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            total_reward = 0.0

            state = obs
            completed = False

            while not completed:
                s = self.get_state_row(state)
                a = self.choose_action(s)

                next_state, reward, terminated, truncated, _ = self.env.step(a)
                self.update_q_value(state, a, reward, next_state, terminated)

                total_reward += reward
                state = next_state
                completed = terminated or truncated

            self.reward_per_episode.append(total_reward)
            self.epsilon_per_episode.append(self.epsilon)
            self.update_epsilon()

    def evaluate(self, n_episodes=10):
        """Run the agent for given number of episodes based on the learned Q-table. Only chooses greedy actions.
        
        Args:
            `n_episodes`: The number of episodes to evaluate for. default = 10
        Returns:
            `total_rewards`: List of total rewards for each episode.
        """
        total_rewards = []
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            total_reward = 0.0

            state = obs
            completed = False

            while not completed:
                s = self.get_state_row(state)
                a = self.greedy_action(s)

                next_state, reward, terminated, truncated, _ = self.env.step(a)

                total_reward += reward
                state = next_state
                completed = terminated or truncated

            total_rewards.append(total_reward)
            
        return total_rewards
