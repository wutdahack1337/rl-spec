import environment
import matplotlib.pyplot as plt
import numpy as np
from argmax import argmax
from tqdm import tqdm


class GreedyAgent:
    def __init__(self, agent_info={}):
        """
        Setup for the agent called when the experiment first starts
        """
        self.num_actions = agent_info.get("num_actions", 2)

        # an array with a count of the number of times each arm has been pulled
        self.arm_count = agent_info.get("arm_count", np.zeros(self.num_actions))

        # the action that the agent took on the previous time step
        # if agent does *no* learning, selects action 0 always
        self.last_action = agent_info.get("last_action", 0)

        # an array with what the agent believes each of the values of the arm are
        self.q_values = agent_info.get("q_values", np.zeros(self.num_actions))

    def get_action(self, reward):
        """
        Takes one step for the agent
        It takes in a reward and observation and returns the action the agent chooses at that time step

        Arguments:
            reward -- float, the reward the agent recieved from the environment after taking the last action

        Returns:
            current_action -- int, the action chosen by the agent at the current time step
        """

        # Q_n = mean(R_t), 1 <= t <= n
        self.arm_count[self.last_action] += 1
        self.q_values[self.last_action] += (
            reward - self.q_values[self.last_action]
        ) / self.arm_count[self.last_action]

        self.last_action = argmax(self.q_values)
        return self.last_action


if __name__ == "__main__":
    num_runs = 200  # The number of times we run the experiment
    num_steps = 1000  # The number of pulls of each arm the agent takes

    env = environment.Environment(
        {"num_arms": 10}
    )  # We set what environment we want to use to test. We pass the enviroment the information it needs. In this case nothing

    rewards = np.zeros((num_runs, num_steps))
    average_best = 0
    for run in tqdm(range(num_runs)):  # tqdm is what creates the progress bar
        np.random.seed(run)

        env.reset()
        agent = GreedyAgent(
            {"num_actions": 10}
        )  # We choose what agent we want to use. We pass the agent the information it needs. Here how many arms there are

        average_best += np.max(env.arms)

        last_reward = 0
        for i in range(num_steps):
            action = agent.get_action(last_reward)
            reward, _ = env.step(action)

            rewards[run, i] = reward

            last_reward = reward

    greedy_scores = np.mean(rewards, axis=0)

    plt.figure(figsize=(15, 5), dpi=80, facecolor="w", edgecolor="k")
    plt.title("Average Reward of Greedy Agent")
    plt.xlabel("Steps")
    plt.ylabel("Average reward")

    plt.plot([average_best / num_runs for _ in range(num_steps)], linestyle="--")
    plt.plot(greedy_scores)
    plt.legend(["Best Possible", "Greedy"])

    plt.savefig("greedy_agent.png")
