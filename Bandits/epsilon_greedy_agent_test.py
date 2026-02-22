import numpy as np

from epsilon_greedy_agent import EpsilonGreedyAgent


def test_1():
    # build a fake agent for testing and set some initial conditions
    np.random.seed(0)
    e_greedy_agent = EpsilonGreedyAgent(
        {
            "q_values": [0, 0, 0.5, 0, 0],
            "arm_count": [0, 1, 0, 0, 0],
            "num_actions": 5,
            "last_action": 1,
            "epsilon": 0.5,
        }
    )

    # given this random seed, we should see a greedy action (action 2) here
    action = e_greedy_agent.get_action(reward=1)

    # make sure the agent is using the argmax that breaks ties randomly
    assert action == 2, "A"

    # make sure to update for the *last_action* not the current action
    assert e_greedy_agent.q_values != [0, 0.5, 1.0, 0, 0], "B"

    # make sure the stepsize is based on the *last_action* not the current action
    assert e_greedy_agent.q_values != [0, 1, 0.5, 0, 0], "C"


def test_2():
    np.random.seed(1)
    e_greedy_agent = EpsilonGreedyAgent(
        {
            "q_values": [0, 0.5, 0.5, 0, 0],
            "arm_count": [0, 1, 0, 0, 0],
            "num_actions": 5,
            "last_action": 1,
            "epsilon": 0.5,
        }
    )

    # given this random seed, we should see a random action (action 4) here
    action = e_greedy_agent.get_action(reward=1)

    # the agent shoud have picked a random action for this particular random seed
    assert action == 4, "A"

    # the agent saw a reward of 1, so should increase the value for *last_action*
    assert e_greedy_agent.q_values == [0, 0.75, 0.5, 0, 0], "B"


def test_3():
    np.random.seed(0)
    e_greedy_agent = EpsilonGreedyAgent(
        {
            "q_values": [0, 0, 1.0, 0, 0],
            "arm_count": [0, 1, 0, 0, 0],
            "num_actions": 5,
            "last_action": 1,
            "epsilon": 0.5,
        }
    )

    action = e_greedy_agent.get_action(reward=1)
    assert e_greedy_agent.q_values == [0, 0.5, 1.0, 0, 0]

    # manipulate the random seed so the agent takes a random action
    np.random.seed(1)

    action = e_greedy_agent.get_action(reward=0)
    assert action == 4

    # check to make sure we update value for action 4
    action = e_greedy_agent.get_action(reward=1)
    assert e_greedy_agent.q_values == [0, 0.5, 0, 0, 1]
