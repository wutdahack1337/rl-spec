import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from epsilon_greedy_agent import EpsilonGreedyAgent
import environment


def averaging_multiple_runs():
    # Why did we averaged over 200 runs?

    num_steps = 1000
    env = environment.Environment({"num_arms": 10})

    plt.figure(figsize=(15, 5), dpi=80, facecolor="w", edgecolor="k")
    plt.title("Comparing three independent runs")
    plt.xlabel("Steps")
    plt.ylabel("Average reward")

    for run in range(3):
        np.random.seed(run)

        averages = []
        env.reset()
        # Plot runs of e-greedy agent
        agent = EpsilonGreedyAgent({"num_actions": 10, "epsilon": 0.1})

        prefix_sum_scores = [0]
        last_reward = 0
        for i in range(num_steps):
            action = agent.get_action(last_reward)
            reward, _ = env.step(action)

            prefix_sum_scores.append(prefix_sum_scores[-1] + reward)
            averages.append(prefix_sum_scores[-1] / (i + 1))

            last_reward = reward

        plt.plot(averages)

    plt.savefig("discussion_runs.png")

    """
    Why does it behave differently in these three runs?

    The answer is that it is due to randomness in the environment and in the agent.

    To compare algorithms, we therefore report performance averaged acroos many runs.
    We do this to ensure that we are not simply reporting a result that is due to stochasticity.
    Rather, we want statistically significant outcomes.
    """


def comparing_values_of_epsilon():
    # Let's try several different values for epsilon and see how they perform

    num_runs = 200
    num_steps = 1000

    env = environment.Environment(({"num_arms": 10}))

    # Experiment code for different e-greedy
    epsilons = [0, 0.01, 0.1, 0.4]

    plt.figure(figsize=(15, 5), dpi=80, facecolor="w", edgecolor="k")
    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.plot([1.55 for _ in range(num_steps)], linestyle="--")

    def smooth(data, window_size):
        return (
            pd.Series(data)
            .rolling(window=window_size, center=True, min_periods=1)
            .mean()
            .values
        )

    for epsilon in epsilons:
        all_rewards = np.zeros((num_runs, num_steps))
        for run in tqdm(range(num_runs)):
            np.random.seed(run)

            env.reset()
            agent = EpsilonGreedyAgent(
                {
                    "num_actions": 10,
                    "epsilon": epsilon,
                }
            )

            last_reward = 0
            for i in range(num_steps):
                action = agent.get_action(last_reward)
                reward, _ = env.step(action)

                all_rewards[run, i] = reward

                last_reward = reward

        scores = np.mean(all_rewards, axis=0)
        plt.plot(smooth(scores, 30))

    plt.legend(["Best Possible"] + epsilons)
    plt.savefig("discussion_epsilons.png")


# TODO
def effect_of_step_size():
    pass


if __name__ == "__main__":
    averaging_multiple_runs()
    comparing_values_of_epsilon()
