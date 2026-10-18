from abc import abstractmethod

import numpy as np
import pandas as pd

from agents.base import EpsilonGreedyAgent
from environment.traffic_light import TrafficLightEnv
from environment.states import LightState, TrafficState, ExtraTimeState
from environment.actions import Actions

class TabularAgent(EpsilonGreedyAgent):
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
        super().__init__(env, discount_factor, initial_epsilon, epsilon_decay, minimum_epsilon, seed)

        # Learning Rate or alpha
        self.lr = learning_rate

        self.n_states = np.prod(self.state_dimensions)

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
    def update_q_value(self, s, action, reward, s_dash, terminated):
        """Update the value in the Q-table based on the agent's algorithm. `s` and `s_dash` are Q table rows."""

    def process_state(self, state):
        """The Q table row of the observation `state`."""
        return self.get_state_row(state)

    def _learn(self, s, action, reward, next_s, terminated):
        """Update the Q table(s) with one transition."""
        self.update_q_value(s, action, reward, next_s, terminated)

    def get_state_row(self, state):
        """Get the row index based on the state (Light, N traffic, S traffic, E traffic, W traffic).

        Args:
            state: The state in the format (Light, N, S, E, W) as returned by `_get_obs()` by the environment.
        Returns:
            `int`: the row index of the state `state` in the Q Table.
        """
        s = [int(x) for x in state]
        return np.ravel_multi_index(s, self.state_dimensions)

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
