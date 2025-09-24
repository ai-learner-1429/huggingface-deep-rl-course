# Utility functions

import numpy as np


def initialize_q_table(state_space: int, action_space: int):
    q_table = np.zeros((state_space, action_space))
    return q_table
