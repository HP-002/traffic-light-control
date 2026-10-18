from agents.base_dqn import DeepNetworkAgent

class DeepQNAgent(DeepNetworkAgent):
    def _compute_targets(self, rewards, next_states, terminated):
        """Returns the TD targets following the DQN algorithm.
        The target network both selects and evaluates the next action.
        y = R + gamma*max_a Q_target(S',a), or y = R if S' is terminal

        Args:
            rewards: tensor of shape `(batch_size,)`
            next_states: encoded next states, tensor of shape `(batch_size, state_size)`
            terminated: 1.0 where the next state is terminal, else 0.0. Tensor of shape `(batch_size,)`
        Returns:
            tensor of shape `(batch_size,)`
        """
        max_values = self.target_network(next_states).max(dim=1).values
        return rewards + self.discount_factor * max_values * (1.0 - terminated)
