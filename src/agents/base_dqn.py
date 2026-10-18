from abc import abstractmethod

import numpy as np
import torch
from torch import nn

from agents.base import EpsilonGreedyAgent
from environment.traffic_light import TrafficLightEnv
from environment.actions import Actions
from network.q_network import QNetwork
from utils.replay_buffer import ReplayBuffer


class DeepNetworkAgent(EpsilonGreedyAgent):
    def __init__(
        self,
        env: TrafficLightEnv,
        learning_rate=1e-3,
        discount_factor=0.95,
        initial_epsilon=1.0,
        epsilon_decay=0.95,
        minimum_epsilon=0.05,
        hidden_sizes=(64, 64),
        buffer_size=10_000,
        batch_size=64,
        learning_starts=100,
        train_frequency=1,
        target_update_frequency=250,
        seed=42,
    ):
        """Initialize the agent.

        Args:
            env: the environment
            learning_rate: learning rate of the Adam optimizer
            discount_factor: gamma
            initial_epsilon: starting epsilon for the epsilon-greedy policy
            epsilon_decay: epsilon is multiplied by this after every episode, starting once learning has started
            minimum_epsilon: lower bound for epsilon
            hidden_sizes: number of units in each hidden layer of the Q network
            buffer_size: maximum transitions kept in the replay buffer
            batch_size: transitions sampled from the replay buffer per update
            learning_starts: transitions to collect before the first update. At least `batch_size` and at most `buffer_size`
            train_frequency: steps between updates of the Q network
            target_update_frequency: steps between copies of the Q network into the target network
            seed: seed for action selection, replay sampling and network initialization
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        if not batch_size <= learning_starts <= buffer_size:
            raise ValueError(
                f"learning_starts must be between batch_size and buffer_size, got batch_size={batch_size}, "
                f"learning_starts={learning_starts}, buffer_size={buffer_size}"
            )
        if train_frequency < 1 or target_update_frequency < 1:
            raise ValueError(
                f"train_frequency and target_update_frequency must be at least 1, "
                f"got {train_frequency} and {target_update_frequency}"
            )

        super().__init__(env, discount_factor, initial_epsilon, epsilon_decay, minimum_epsilon, seed)

        # Learning Rate of the optimizer
        self.lr = learning_rate

        # Each dimension is one-hot encoded and concatenated, so the network input has size 9 + 4*4 + 2 = 27.
        # state_offsets[i] is the position of the first entry of dimension i in the encoded vector.
        self.state_size = sum(self.state_dimensions)
        self.state_offsets = np.cumsum((0,) + self.state_dimensions[:-1])

        # Q network is trained, target network is a periodic copy of it used for the targets.
        # The weights are initialized from the seed without changing torch's global random state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.q_network = QNetwork(self.state_size, self.n_actions, hidden_sizes)
            self.target_network = QNetwork(self.state_size, self.n_actions, hidden_sizes)
        self.update_target_network()
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss()

        # Experience replay
        self.replay_buffer = ReplayBuffer(buffer_size, self.state_size, self.rng)
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.train_frequency = train_frequency
        self.target_update_frequency = target_update_frequency

        # Total environment steps taken while training
        self.n_steps = 0

        # Log
        self.loss_per_episode = []
        # Losses of the updates in the current episode
        self.episode_losses = []

    @abstractmethod
    def _compute_targets(self, rewards, next_states, terminated):
        """Returns the TD targets for a batch of transitions based on the agent's algorithm.

        Args:
            rewards: tensor of shape `(batch_size,)`
            next_states: encoded next states, tensor of shape `(batch_size, state_size)`
            terminated: 1.0 where the next state is terminal, else 0.0. Tensor of shape `(batch_size,)`
        Returns:
            tensor of shape `(batch_size,)`
        """

    @property
    def learning_started(self):
        """True once `learning_starts` transitions have been collected and the Q network is being updated."""
        return self.n_steps >= self.learning_starts

    def update_target_network(self):
        """Copy the weights of the Q network into the target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def encode_state(self, state):
        """One-hot encode each dimension of the state and concatenate them.

        Args:
            state: The state in the format (Light, N, S, E, W, Extra time) as returned by `_get_obs()` by the environment.
        Returns:
            `np.ndarray`: of size `state_size` with a single 1 per dimension.
        """
        encoded = np.zeros(self.state_size, dtype=np.float32)
        encoded[self.state_offsets + np.asarray(state, dtype=np.int64)] = 1.0
        return encoded

    def process_state(self, state):
        """The one-hot encoding of the observation `state`, see `encode_state`."""
        return self.encode_state(state)

    def get_loss_per_episode(self):
        """Returns mean loss per episode.

        Returns:
            `list`: of size `n_episodes` with the mean loss of the updates in that episode, NaN if there were none.
        """
        return self.loss_per_episode

    def print_q_network_architecture(self):
        """Print the layers of the Q network along with its input size, output actions and parameter count.
        The target network has the same architecture.
        """
        n_params = sum(p.numel() for p in self.q_network.parameters())
        n_trainable = sum(p.numel() for p in self.q_network.parameters() if p.requires_grad)
        actions = [Actions(a).name for a in range(self.n_actions)]

        print("Q Network:")
        print(self.q_network)
        print(f"Input size: {self.state_size} (one-hot encoded state)")
        print(f"Output size: {self.n_actions} {actions}")
        print(f"Parameters: {n_params} ({n_trainable} trainable)")

    def save_q_network(self, filename):
        """Save the `state_dict` of the Q network to `filename`.
        Load it with `agent.q_network.load_state_dict(torch.load(filename))` on an agent created with the same `hidden_sizes`.
        To resume training later, use `save_checkpoint` instead.
        """
        torch.save(self.q_network.state_dict(), filename)

    def save_checkpoint(self, filename):
        """Save everything needed to resume training to `filename`: both networks, the optimizer, the replay buffer,
        epsilon, the step count, the random generator and the logs. The environment's random state is not included.
        """
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "replay_buffer": self.replay_buffer.state_dict(),
                "epsilon": self.epsilon,
                "n_steps": self.n_steps,
                "rng": self.rng.bit_generator.state,
                "reward_per_episode": self.reward_per_episode,
                "epsilon_per_episode": self.epsilon_per_episode,
                "loss_per_episode": self.loss_per_episode,
            },
            filename,
        )

    def load_checkpoint(self, filename):
        """Restore a checkpoint written by `save_checkpoint` into an agent created with the same
        `hidden_sizes` and `buffer_size`.
        """
        checkpoint = torch.load(filename, weights_only=True)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.replay_buffer.load_state_dict(checkpoint["replay_buffer"])
        self.epsilon = checkpoint["epsilon"]
        self.n_steps = checkpoint["n_steps"]
        self.rng.bit_generator.state = checkpoint["rng"]
        self.reward_per_episode = checkpoint["reward_per_episode"]
        self.epsilon_per_episode = checkpoint["epsilon_per_episode"]
        self.loss_per_episode = checkpoint["loss_per_episode"]

    def get_action_values(self, s):
        """Returns the values of the actions for the encoded state `s` from the Q network"""
        x = torch.as_tensor(s).unsqueeze(0)
        with torch.no_grad():
            return self.q_network(x).squeeze(0).numpy()

    def greedy_action(self, s):
        """Choose the greedy action for the encoded state `s`.
        Exact ties between network outputs practically never happen, so no random tie-breaking is needed.

        Returns
            int: The best action to take for the given state `s`
        """
        return int(self.get_action_values(s).argmax())

    def update_network(self, s, action, reward, next_s, terminated):
        """Store the transition in the replay buffer, then
        - every `train_frequency` steps, once `learning_starts` transitions are stored, update the Q network on a sampled batch
        - every `target_update_frequency` steps, copy the Q network into the target network

        Args:
            s: Encoded current state.
            action: Action taken from current state.
            reward: Immediate reward recieved.
            next_s: Encoded next state reached.
            terminated: True if next state is terminal, else False.
        Returns:
            `float | None`: the loss if the Q network was updated, else None.
        """
        self.replay_buffer.add(s, int(action), reward, next_s, terminated)
        self.n_steps += 1

        loss = None
        if self.learning_started and self.n_steps % self.train_frequency == 0:
            loss = self._optimize()

        if self.n_steps % self.target_update_frequency == 0:
            self.update_target_network()

        return loss

    def _optimize(self):
        """Take one gradient step on a batch sampled from the replay buffer.

        Returns:
            `float`: the loss of the batch
        """
        states, actions, rewards, next_states, terminated = (
            torch.as_tensor(x) for x in self.replay_buffer.sample(self.batch_size)
        )

        # Q(S,A) for the actions that were taken
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            targets = self._compute_targets(rewards, next_states, terminated)

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _learn(self, s, action, reward, next_s, terminated):
        """Store the transition and update the networks, keeping the loss for the episode's log."""
        loss = self.update_network(s, action, reward, next_s, terminated)
        if loss is not None:
            self.episode_losses.append(loss)

    def _end_episode(self):
        """Log the mean loss of the episode and decay epsilon, but only once learning has started
        so that the exploration budget is not spent while the network is still untrained.
        """
        self.loss_per_episode.append(float(np.mean(self.episode_losses)) if self.episode_losses else float("nan"))
        self.episode_losses = []
        if self.learning_started:
            self.update_epsilon()

    def train(self, n_episodes=10):
        """Train the agent for given number of episodes.
        Repeated calls continue training and append to the reward, epsilon and loss logs.

        Args:
            `n_episodes`: The number of episodes to train for. default = 10
        """
        self.q_network.train()
        super().train(n_episodes)

    def evaluate(self, n_episodes=10):
        """Run the agent for given number of episodes based on the learned Q network. Only chooses greedy actions.

        Args:
            `n_episodes`: The number of episodes to evaluate for. default = 10
        Returns:
            `total_rewards`: List of total rewards for each episode.
        """
        self.q_network.eval()
        return super().evaluate(n_episodes)
