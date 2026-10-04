import numpy as np

from agents.base import Agent

class QLearningAgent(Agent):
    def _init_tables(self):
        self.Q = np.zeros((self.n_states, self.n_actions))

    def get_q_tables(self):
        return self.Q, None, False

    def _get_action_values(self, s):
        return self.Q[s]
    
    def update_q_value(self, state, action, reward, next_state):
        s_dash = self.get_state_row(next_state)
        action_values = self._get_action_values(s_dash)
        max_value = action_values[np.argmax(action_values)]

        s = self.get_state_row(state)
        a = int(action)

        # Q(S,A) <- Q(S,A) + alpha*[R + gamma*max_a Q(S,A) - Q(S,A)]
        self.Q[s][a] += self.lr * (reward + self.discount_factor * max_value - self.Q[s][a])

        return next_state

    def greedy_action(self, s):
        action_values = self._get_action_values(s)
        return np.argmax(action_values)

    def choose_action(self, s):
        if self.rng.random() <= self.epsilon:
            return self.rng.integers(self.n_actions)
        return self.greedy_action(s)
    
    def train(self, n_episodes=10):
        self.reward_per_episode = []
        self.epsilon_per_episode = []
        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            total_reward = 0.0

            state = obs
            completed = False

            while not completed:
                s = self.get_state_row(state)
                a = self.choose_action(s)

                next_state, reward, terminated, truncated, _ = self.env.step(a)
                self.update_q_value(state, a, reward, next_state)

                total_reward += reward
                state = next_state
                completed = terminated or truncated

            self.reward_per_episode.append(total_reward)
            self.epsilon_per_episode.append(self.epsilon)
            self.update_epsilon()

    def evaluate(self, n_episodes=10):
        total_rewards = []
        for episode in range(n_episodes):
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
