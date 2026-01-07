import numpy as np


def argmax(q_values):
    """
    Takes in a list of q_values and returns the index of the item with the highest value
    Breaks ties randomly

    returns: int - the index of the highest value in q_values
    """

    mx = -float("inf")
    res = []
    for i in range(len(q_values)):
        if q_values[i] > mx:
            mx = q_values[i]
            res = [i]
        elif q_values[i] == mx:
            res.append(i)

    return np.random.choice(res)
