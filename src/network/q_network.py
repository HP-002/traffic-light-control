from torch import nn

class QNetwork(nn.Module):
    """Fully connected network that maps an encoded state to the Q-values of every action."""

    def __init__(self, input_size, n_actions, hidden_sizes=(64, 64)):
        """Build the network.

        Args:
            input_size: size of the encoded state vector
            n_actions: number of actions, one output per action
            hidden_sizes: number of units in each hidden layer
        """
        super().__init__()

        layers = []
        in_size = input_size
        for hidden_size in hidden_sizes:
            layers += [nn.Linear(in_size, hidden_size), nn.ReLU()]
            in_size = hidden_size
        layers.append(nn.Linear(in_size, n_actions))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """Returns the Q-values for a batch of encoded states.

        Args:
            x: tensor of shape `(batch_size, input_size)`
        Returns:
            tensor of shape `(batch_size, n_actions)`
        """
        return self.layers(x)
