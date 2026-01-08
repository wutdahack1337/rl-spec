import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import environment
from argmax import argmax
from greedy_agent import GreedyAgent


class EpsilonGreedyAgent:
    def __init__(self, agent_info={}):
        self.num_actions = agent_info.get("num_actions", 2)
        self.q_values = agent_info.get("q_values", np.zeros(self.num_actions))
        self.arm_count = agent_info.get("arm_count", np.zeros(self.num_actions))
        self.last_action = agent_info.get("last_action", 0)
        self.epsilon = agent_info.get("epsilon", 0.1)

    def get_action(self, reward):
        """
        Takes one step for the agent. It takes in a reward and observation and returns the action the agent chooses at that time step

        Arguments:
            reward -- float, the reward the agent recieved from the environment after taking the last action

        Returns:
            current_action -- int, the action chosen by the agent at the current time step
        """

        self.arm_count[self.last_action] += 1
        self.q_values[self.last_action] += (
            reward - self.q_values[self.last_action]
        ) / self.arm_count[self.last_action]

        if np.random.random() < self.epsilon:
            self.last_action = np.random.randint(self.num_actions)
        else:
            self.last_action = argmax(self.q_values)

        return self.last_action


if __name__ == "__main__":
    num_runs = 200
    num_steps = 1000

    env = environment.Environment(({"num_arms": 10}))

    ######### ===  Epsilon Greedy Agent === #########
    all_rewards = np.zeros((num_runs, num_steps))
    average_best = 0
    for run in tqdm(range(num_runs)):
        np.random.seed(run)

        env.reset()
        agent = EpsilonGreedyAgent(
            {
                "num_actions": 10,
                "epsilon": 0.1,
            }
        )

        average_best += np.max(env.arms)

        last_reward = 0
        for i in range(num_steps):
            action = agent.get_action(last_reward)
            reward, _ = env.step(action)

            all_rewards[run, i] = reward

            last_reward = reward

    scores = np.mean(all_rewards, axis=0)

    plt.figure(figsize=(15, 5), dpi=80, facecolor="w", edgecolor="k")
    plt.title("Average Reward of Greedy Agent vs. E-Greedy Agent")
    plt.xlabel("Steps")
    plt.ylabel("Average reward")

    plt.plot([average_best / num_runs for _ in range(num_steps)], linestyle="--")
    plt.plot(scores)

    ######## ===  Greedy Agent === #########
    all_rewards = np.zeros((num_runs, num_steps))
    for run in tqdm(range(num_runs)):
        np.random.seed(run)

        env.reset()
        agent = GreedyAgent(
            {
                "num_actions": 10,
            }
        )

        last_reward = 0
        for i in range(num_steps):
            action = agent.get_action(last_reward)
            reward, _ = env.step(action)

            all_rewards[run, i] = reward

            last_reward = reward

    scores = np.mean(all_rewards, axis=0)

    plt.plot(scores)
    plt.legend(("Best Possible", "Greedy", "Epsilon: 0.1"))
    plt.savefig("epsilon_greedy_agent.png")
