import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from loguru import logger


class Bandit(ABC):

    """
    Abstract base class for multi-armed bandit algorithms.

    Attributes:
        p (list): Probabilities or expected rewards for each bandit.
    """

    @abstractmethod
    def __init__(self, p):
        """
        Initialize the Bandit with given probabilities.

        Args:
            p (list): Expected reward for each bandit arm.
        """
        pass

    @abstractmethod
    def __repr__(self):
        """Return a string representation of the bandit object."""
        pass

    @abstractmethod
    def pull(self):
        """Select an arm to pull."""
        pass

    @abstractmethod
    def update(self):
        """Update internal state with the reward received."""
        pass

    @abstractmethod
    def experiment(self):
        """Run the experiment for a given number of trials."""
        pass

    @abstractmethod
    def report(self):
        """Generate a report: save data, log total reward and regret."""
        pass



class EpsilonGreedy(Bandit):

    """
    Epsilon-Greedy algorithm with decaying epsilon strategy.
    """

    def __init__(self, p):
        """Initialize values and counters."""
        self.p = p
        self.counts = np.zeros(len(p))
        self.values = np.zeros(len(p))
        self.total_reward = 0
        self.regret = 0
        self.history = []

    def __repr__(self):
        """String representation."""
        return f"EpsilonGreedy(p={self.p})"

    def pull(self, epsilon):
        """Choose arm using epsilon-greedy policy."""
        if np.random.rand() < epsilon:
            return np.random.randint(0, len(self.p))
        else:
            return np.argmax(self.values)

    def update(self, arm, reward):
        """Update estimated values and stats."""
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]
        self.total_reward += reward
        self.regret += max(self.p) - reward
        self.history.append((arm, reward, "EpsilonGreedy"))

    def experiment(self, trials):
        """Run epsilon-greedy strategy for given number of trials."""
        for t in range(1, trials+1):
            epsilon = 1 / t
            arm = self.pull(epsilon)
            reward = np.random.normal(self.p[arm])
            self.update(arm, reward)

    def report(self):
        """Save results to CSV and log final stats."""
        df = pd.DataFrame(self.history, columns=["Bandit", "Reward", "Algorithm"])
        df.to_csv("epsilon_greedy_rewards.csv", index=False)
        logger.info(f"[EpsilonGreedy] Total Reward: {self.total_reward}")
        logger.info(f"[EpsilonGreedy] Total Regret: {self.regret}")


class ThompsonSampling(Bandit):

    """
    Thompson Sampling algorithm assuming binary rewards.
    """

    def __init__(self, p):
        """Initialize priors and stats."""
        self.p = p
        self.alpha = np.ones(len(p))
        self.beta = np.ones(len(p))
        self.total_reward = 0
        self.regret = 0
        self.history = []

    def __repr__(self):
        """String representation."""
        return f"ThompsonSampling(p={self.p})"

    def pull(self):
        """Sample from beta distribution and pick arm."""
        sampled_theta = np.random.beta(self.alpha, self.beta)
        return np.argmax(sampled_theta)

    def update(self, arm, reward):
        """Update beta distribution and stats based on binary reward."""
        binary_reward = 1 if reward > np.mean(self.p) else 0
        self.alpha[arm] += binary_reward
        self.beta[arm] += 1 - binary_reward
        self.total_reward += reward
        self.regret += max(self.p) - reward
        self.history.append((arm, reward, "ThompsonSampling"))

    def experiment(self, trials):
        """Run Thompson Sampling for a given number of trials."""
        for _ in range(trials):
            arm = self.pull()
            reward = np.random.normal(self.p[arm])
            self.update(arm, reward)

    def report(self):
        """Save results to CSV and log final stats."""
        df = pd.DataFrame(self.history, columns=["Bandit", "Reward", "Algorithm"])
        df.to_csv("thompson_sampling_rewards.csv", index=False)
        logger.info(f"[ThompsonSampling] Total Reward: {self.total_reward}")
        logger.info(f"[ThompsonSampling] Total Regret: {self.regret}")


class Visualization:

    """
    Class for plotting learning curves and cumulative rewards.
    """

    @staticmethod
    def plot_learning(history, name):
        """Plot average arm index over time."""
        plt.figure()
        arms, _, _ = zip(*history)
        plt.plot(np.cumsum(arms) / (np.arange(len(arms)) + 1))
        plt.title(f"{name} - Learning Curve")
        plt.xlabel("Trials")
        plt.ylabel("Average Arm Index")
        plt.grid()
        plt.show()

    @staticmethod
    def plot_cumulative_rewards(eg_history, ts_history):
        """Plot cumulative rewards for both algorithms."""
        plt.figure()
        plt.plot(np.cumsum([r for _, r, _ in eg_history]), label="Epsilon-Greedy")
        plt.plot(np.cumsum([r for _, r, _ in ts_history]), label="Thompson Sampling")
        plt.title("Cumulative Rewards")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Reward")
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == "__main__":
    Bandit_Reward = [1, 2, 3, 4]
    trials = 20000

    eg = EpsilonGreedy(Bandit_Reward)
    eg.experiment(trials)
    eg.report()

    ts = ThompsonSampling(Bandit_Reward)
    ts.experiment(trials)
    ts.report()

    Visualization.plot_learning(eg.history, "Epsilon-Greedy")
    Visualization.plot_learning(ts.history, "Thompson Sampling")
    Visualization.plot_cumulative_rewards(eg.history, ts.history)