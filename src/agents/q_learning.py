import numpy as np

from agents.base import Agent

class QLearningAgent(Agent):
    def _init_tables(self):
        self.Q = np.zeros((self.n_states, self.n_actions))

    def get_q_tables(self):
        return self.Q, None, False

    def _get_action_values(self, s):
        return self.Q[s]
    
    def update_q_value(self, state, action, reward, next_state, terminated):
        s_dash = self.get_state_row(next_state)
        action_values = self._get_action_values(s_dash)
        max_value = action_values[np.argmax(action_values)]

        s = self.get_state_row(state)
        a = int(action)

        # Q(S,A) <- Q(S,A) + alpha*[R + gamma*max_a Q(S',A) - Q(S,A)]
        if terminated:
            target = reward
        else:
            target = reward + self.discount_factor * max_value
        self.Q[s][a] += self.lr * (target - self.Q[s][a])

        return next_state
