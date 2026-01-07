import numpy as np
from greedy_agent import GreedyAgent


def test_0():
    # build a fake agent for testing and set some initial conditions
    np.random.seed(1)
    greedy_agent = GreedyAgent(
        {
            "q_values": [0, 0, 0.5, 0, 0],
            "arm_count": [0, 1, 0, 0, 0],
            "last_action": 1,
        }
    )

    # take a fake agent step
    action = greedy_agent.get_action(reward=1)

    # make sure the q_values were updated correctly
    assert greedy_agent.q_values == [0, 0.5, 0.5, 0, 0]

    # make sure the agent is using the argmax that breaks ties randomly
    assert action == 2


def test_1():
    np.random.seed(1)
    greedy_agent = GreedyAgent(
        {
            "q_values": [0, 0, 1.0, 0, 0],
            "arm_count": [0, 1, 0, 0, 0],
            "last_action": 1,
        }
    )

    # take a fake agent step
    action = greedy_agent.get_action(reward=1)

    # make sure agent took greedy action
    assert action == 2

    # make sure the q_values were updated correctly
    assert greedy_agent.q_values == [0, 0.5, 1.0, 0, 0]

    # take another step
    action = greedy_agent.get_action(reward=2)
    assert action == 2
    assert greedy_agent.q_values == [0, 0.5, 2.0, 0, 0]
