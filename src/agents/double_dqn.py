from agents.base_dqn import DeepNetworkAgent

class DoubleDQNAgent(DeepNetworkAgent):
    def _compute_targets(self, rewards, next_states, terminated):
        """Returns the TD targets following the Double DQN algorithm.
        The Q network selects the next action and the target network evaluates it, which reduces the
        overestimation caused by taking the max over noisy estimates.
        y = R + gamma*Q_target(S', argmax_a Q(S',a)), or y = R if S' is terminal

        Args:
            rewards: tensor of shape `(batch_size,)`
            next_states: encoded next states, tensor of shape `(batch_size, state_size)`
            terminated: 1.0 where the next state is terminal, else 0.0. Tensor of shape `(batch_size,)`
        Returns:
            tensor of shape `(batch_size,)`
        """
        max_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
        max_values = self.target_network(next_states).gather(1, max_actions).squeeze(1)
        return rewards + self.discount_factor * max_values * (1.0 - terminated)
