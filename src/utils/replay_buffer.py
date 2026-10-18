import numpy as np
import torch

class ReplayBuffer:
    """Fixed size circular buffer of transitions. Once full, the oldest transition is overwritten."""

    def __init__(self, capacity, state_size, rng):
        """Create an empty buffer.

        Args:
            capacity: maximum number of transitions stored
            state_size: size of the encoded state vector
            rng: numpy random generator used for sampling
        """
        if capacity < 1:
            raise ValueError(f"capacity must be at least 1, got {capacity}")
        self.capacity = capacity
        self.rng = rng

        self.states = np.zeros((capacity, state_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_size), dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)

        # Position of the next write and number of transitions stored
        self.index = 0
        self.size = 0

    def __len__(self):
        return self.size

    def add(self, state, action, reward, next_state, terminated):
        """Store one transition. `state` and `next_state` must already be encoded."""
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.terminated[self.index] = terminated

        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Sample `batch_size` transitions uniformly without replacement.

        Returns:
            `tuple`: (states, actions, rewards, next_states, terminated) as numpy arrays
        """
        if batch_size > self.size:
            raise ValueError(f"cannot sample {batch_size} transitions, only {self.size} stored")
        idx = self.rng.choice(self.size, size=batch_size, replace=False)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.terminated[idx],
        )

    def state_dict(self):
        """Returns the stored transitions and write position as tensors, for saving with `torch.save`."""
        return {
            "states": torch.from_numpy(self.states),
            "actions": torch.from_numpy(self.actions),
            "rewards": torch.from_numpy(self.rewards),
            "next_states": torch.from_numpy(self.next_states),
            "terminated": torch.from_numpy(self.terminated),
            "index": self.index,
            "size": self.size,
        }

    def load_state_dict(self, state_dict):
        """Restore the buffer from `state_dict()`. The capacity and state size must match."""
        for name in ("states", "actions", "rewards", "next_states", "terminated"):
            getattr(self, name)[...] = state_dict[name].numpy()
        self.index = state_dict["index"]
        self.size = state_dict["size"]
