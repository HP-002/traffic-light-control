import json
from abc import ABC, abstractmethod

import numpy as np

from environment.traffic_light import TrafficLightEnv

class EpsilonGreedyAgent(ABC):
    """Shared base of the tabular and deep agents: epsilon-greedy action selection and the train/evaluate loops.

    Subclasses convert observations into their own state representation with `process_state`
    (a Q table row or an encoded vector) and learn from transitions in `_learn`.
    """

    def __init__(
        self,
        env: TrafficLightEnv,
        discount_factor=0.95,
        initial_epsilon=1.0,
        epsilon_decay=0.95,
        minimum_epsilon=0.05,
        seed=42,
    ):
        # Environment
        self.env = env

        # Discount Factor or gamma
        self.discount_factor = discount_factor

        # Epsilon - Used for epsilon-greedy algorithm
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = minimum_epsilon

        # Seed for randomness
        self.seed = seed
        self.rng = np.random.default_rng(np.random.SeedSequence(self.seed).spawn(1)[0])

        # Observation space is [Light, N traffic, S traffic, E traffic, W traffic, Extra time].
        # The dimensions of each are [9, 4, 4, 4, 4, 2]
        self.state_dimensions = tuple(int(n) for n in env.observation_space.nvec) # (9, 4, 4, 4, 4, 2)
        self.n_actions = env.action_space.n

        # Log
        self.reward_per_episode = []
        self.epsilon_per_episode = []
        # Traffic metrics per episode, see `TrafficLightEnv.get_episode_metrics`
        self.metrics_per_episode = self._empty_metrics()
        # Rewards and traffic metrics of the most recent call to `evaluate`
        self.evaluation_reward_per_episode = []
        self.evaluation_metrics_per_episode = self._empty_metrics()

    @abstractmethod
    def process_state(self, state):
        """Convert an observation from the environment into the representation used by the agent."""

    @abstractmethod
    def get_action_values(self, s):
        """Returns the values of the actions for the processed state `s`"""

    @abstractmethod
    def _learn(self, s, action, reward, next_s, terminated):
        """Learn from one transition between the processed states `s` and `next_s`."""

    def _end_episode(self):
        """Called after every training episode, once the reward and epsilon have been logged."""
        self.update_epsilon()

    def update_epsilon(self):
        """Update epsilon value based on decay.
        New epsilon value is `max(min_epsilon, epsilon_decay * epsilon)`
        """
        self.epsilon = max(self.min_epsilon, self.epsilon_decay * self.epsilon)

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

    def _empty_metrics(self):
        """Returns a dict with an empty list for each traffic metric in `TrafficLightEnv.METRIC_NAMES`."""
        return {name: [] for name in self.env.METRIC_NAMES}

    def _log_metrics(self, metrics_per_episode):
        """Append the traffic metrics of the episode that just ended to `metrics_per_episode`."""
        for name, value in self.env.get_episode_metrics().items():
            metrics_per_episode[name].append(value)

    def get_metrics_per_episode(self):
        """Returns the traffic metrics per training episode.

        Returns:
            `dict`: metric name (see `TrafficLightEnv.get_episode_metrics`) to a list of size `n_episodes`.
        """
        return self.metrics_per_episode

    def get_evaluation_metrics_per_episode(self):
        """Returns the traffic metrics per episode of the most recent call to `evaluate`.

        Returns:
            `dict`: metric name (see `TrafficLightEnv.get_episode_metrics`) to a list with one value per episode.
        """
        return self.evaluation_metrics_per_episode

    def get_hyperparameters(self):
        """Returns the hyperparameters the agent was created with.

        Returns:
            `dict`: hyperparameter name to value. Subclasses add their own to the dict returned by `super()`.
        """
        return {
            "discount_factor": self.discount_factor,
            "initial_epsilon": self.initial_epsilon,
            "epsilon_decay": self.epsilon_decay,
            "minimum_epsilon": self.min_epsilon,
            "seed": self.seed,
        }

    def get_results(self):
        """Returns the hyperparameters and the training and evaluation logs of the agent.

        Returns:
            `dict`: Subclasses add their own logs to the dict returned by `super()`.
        """
        return {
            "agent": type(self).__name__,
            "stochastic": self.env.stochastic,
            "hyperparameters": self.get_hyperparameters(),
            "n_episodes": len(self.reward_per_episode),
            "reward_per_episode": [float(r) for r in self.reward_per_episode],
            "epsilon_per_episode": [float(e) for e in self.epsilon_per_episode],
            "metrics_per_episode": {
                name: [float(v) for v in values] for name, values in self.metrics_per_episode.items()
            },
            "evaluation_reward_per_episode": [float(r) for r in self.evaluation_reward_per_episode],
            "evaluation_metrics_per_episode": {
                name: [float(v) for v in values] for name, values in self.evaluation_metrics_per_episode.items()
            },
        }

    def save_results(self, filename):
        """Save the results returned by `get_results` to `filename` as JSON.
        Load them with `json.load` to compare agents without retraining.
        """
        with open(filename, "w") as f:
            json.dump(self.get_results(), f, indent=2)

    def greedy_action(self, s):
        """Choose the greedy action for the processed state s

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
            int: An action to take for the processed state `s`
        """
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.greedy_action(s)

    def train(self, n_episodes=10):
        """Train the agent for given number of episodes.
        Repeated calls continue training and append to the logs.

        Args:
            `n_episodes`: The number of episodes to train for. default = 10
        """
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            total_reward = 0.0

            s = self.process_state(obs)
            completed = False

            while not completed:
                a = self.choose_action(s)

                next_state, reward, terminated, truncated, _ = self.env.step(a)
                next_s = self.process_state(next_state)
                self._learn(s, a, reward, next_s, terminated)

                total_reward += reward
                s = next_s
                completed = terminated or truncated

            self.reward_per_episode.append(total_reward)
            self.epsilon_per_episode.append(self.epsilon)
            self._log_metrics(self.metrics_per_episode)
            self._end_episode()

    def evaluate(self, n_episodes=10):
        """Run the agent for given number of episodes based on what it has learned. Only chooses greedy actions.
        The rewards and traffic metrics are also kept in `evaluation_reward_per_episode` and
        `evaluation_metrics_per_episode`, replacing those of the previous call.

        Args:
            `n_episodes`: The number of episodes to evaluate for. default = 10
        Returns:
            `total_rewards`: List of total rewards for each episode.
        """
        total_rewards = []
        metrics = self._empty_metrics()
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            total_reward = 0.0

            s = self.process_state(obs)
            completed = False

            while not completed:
                a = self.greedy_action(s)

                next_state, reward, terminated, truncated, _ = self.env.step(a)

                total_reward += reward
                s = self.process_state(next_state)
                completed = terminated or truncated

            total_rewards.append(total_reward)
            self._log_metrics(metrics)

        self.evaluation_reward_per_episode = total_rewards
        self.evaluation_metrics_per_episode = metrics
        return total_rewards
