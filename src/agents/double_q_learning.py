import numpy as np

from agents.base import Agent

class DoubleQLearningAgent(Agent):
    def _init_tables(self):
        """Initialize both the Q tables"""
        self.Q1 = np.zeros((self.n_states, self.n_actions))
        self.Q2 = np.zeros((self.n_states, self.n_actions))

    def get_q_tables(self):
        """Returns both the Q-tables.
        Returns:
            Q1: Q-Table 1
            Q2: Q-Table 2
            True: For Double Q-Learning Algorithm
        """
        return self.Q1, self.Q2, True

    def get_action_values(self, s):
        """Returns the sum of values for all the actions given the current state from both the tables
        
        Returns:
            `list`: Of size `action_space` with the sum of Q-values from both tables for all actions for the state
        """
        return self.Q1[s] + self.Q2[s]
    
    def update_q_value(self, state, action, reward, next_state, terminated):
        """Update value for the given state-action pair based on the reward and next state following the Double Q-Learning algorithm.
        - With Probability 0.5: Q1(S,A) <- Q1(S,A) + alpha*[R + gamma*Q2(S',max_a Q1(S',A)) - Q1(S,A)]
        - with Probability 0.5: Q2(S,A) <- Q2(S,A) + alpha*[R + gamma*Q1(S',max_a Q2(S',A)) - Q2(S,A)]

        Args:
            state: Current State.
            action: Action taken from current state.
            reward: Immediate reward recieved.
            next_state: The next state reached.
            terminated: True if next state is terminal, else False.
        """
        s_dash = self.get_state_row(next_state)
        s = self.get_state_row(state)
        a = int(action)

        if self.rng.random() < 0.5:
            # Q1(S,A) <- Q1(S,A) + alpha*[R + gamma*Q2(S',max_a Q1(S',A)) - Q1(S,A)]
            max_action = np.argmax(self.Q1[s_dash])
            max_value = self.Q2[s_dash][max_action]
            if terminated:
                target = reward
            else:
                target = reward + self.discount_factor * max_value
            self.Q1[s][a] += self.lr * (target - self.Q1[s][a])
        else:
            # Q2(S,A) <- Q2(S,A) + alpha*[R + gamma*Q1(S',max_a Q2(S',A)) - Q2(S,A)]
            max_action = np.argmax(self.Q2[s_dash])
            max_value = self.Q1[s_dash][max_action]
            if terminated:
                target = reward
            else:
                target = reward + self.discount_factor * max_value
            self.Q2[s][a] += self.lr * (target - self.Q2[s][a])
