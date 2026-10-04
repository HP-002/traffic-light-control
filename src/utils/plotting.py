import numpy as np
import matplotlib.pyplot as plt

def plot_reward_per_episode(reward_per_episode: list):
    plt.plot(range(1, len(reward_per_episode) + 1), reward_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward per Episode")
    plt.title("Total Reward per Episode")
    plt.show()


def plot_epsilon_per_episode(epsilon_per_episode: list, initial_epsilon, minimum_epsilon, epsilon_decay):
    label = f"initial_epsilon={initial_epsilon}, minimum_epsilon={minimum_epsilon}, epsilon_decay={epsilon_decay}"
    plt.plot(range(1, len(epsilon_per_episode) + 1), epsilon_per_episode, label=label)
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Epsilon per Episode")
    plt.legend()
    plt.show()