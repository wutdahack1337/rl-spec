import numpy as np

from argmax import argmax


def test_0():
    test_array = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    assert argmax(test_array) == 8, (
        "Check your argmax implement returns the index of the largest value"
    )

    # makesure np.random.choice is called correctly
    np.random.seed(0)
    test_array = [1, 0, 0, 1]


def test_1():
    # set random seed so results are deterministic
    np.random.seed(0)
    test_array = [1, 0, 0, 1]

    counts = [0, 0, 0, 0]
    for _ in range(100):
        counts[argmax(test_array)] += 1

    # make sure argmax does not always choose first entry
    assert counts[0] != 100, (
        "Make sure your argmax implementation randomly chooses among the largest value"
    )

    # make sure argmax does not always choose last entry
    assert counts[-1] != 100, (
        "Make sure your argmax implementation radomly chooses among the largest value"
    )

    # make sure the random number generator is called exactly once whenver `argmax` is called
    expected = [44, 0, 0, 56]  # <--- notice not perfectly uniform due to randomness
    assert counts == expected
