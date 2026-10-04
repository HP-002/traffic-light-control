import numpy as np

from agents.base import Agent

class QLearningAgent(Agent):
    def _init_tables(self):
        """Initialize the Q Table"""
        self.Q = np.zeros((self.n_states, self.n_actions))

    def get_q_tables(self):
        """Returns the Q table.
        Returns:
            Q: The Q table
            None: Not required for Q-Learning Algorithm
            False: For Q-Learning Algorithm
        """
        return self.Q, None, False

    def get_action_values(self, s):
        """Returns the values for all the actions given the current state
        
        Returns:
            `list`: Of size `action_space` with the Q-values for all actions for the state
        """
        return self.Q[s]
    
    def update_q_value(self, state, action, reward, next_state, terminated):
        """Update value for the given state-action pair based on the reward and next state following the Q-Learning algorithm.
        Q(S,A) <- Q(S,A) + alpha*[R + gamma*max_a Q(S',A) - Q(S,A)]

        Args:
            state: Current State.
            action: Action taken from current state.
            reward: Immediate reward recieved.
            next_state: The next state reached.
            terminated: True if next state is terminal, else False.
        """

        s_dash = self.get_state_row(next_state)
        action_values = self.get_action_values(s_dash)
        max_value = action_values[np.argmax(action_values)]

        s = self.get_state_row(state)
        a = int(action)

        # Q(S,A) <- Q(S,A) + alpha*[R + gamma*max_a Q(S',A) - Q(S,A)]
        if terminated:
            target = reward
        else:
            target = reward + self.discount_factor * max_value
        self.Q[s][a] += self.lr * (target - self.Q[s][a])
