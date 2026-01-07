import numpy as np


class Environment:
    def __init__(self, env_info={}):
        """
        Setup for the environment called when the experiment first starts

        Note:
            Initialize a tuple with the reward, first state observation, boolean indicating if it's terminal
        """

        self.num_arms = env_info.get("num_arms", 2)

        # self.arms[i] <=> mean
        self.arms = np.random.randn(
            self.num_arms
        )  # [np.random.normal(0.0, 1.0) for _ in range(self.num_arms)]

    def reset(self):
        self.arms = np.random.randn(self.num_arms)

    def step(self, action):
        """
        A step taken by the environment

        Args:
            action: The action taken by the agent

        Returns:
            (float, Boolean): a tuple of the reward, and boolean indicating if it's terminal
        """

        reward = self.arms[action] + np.random.randn()
        return reward, False
