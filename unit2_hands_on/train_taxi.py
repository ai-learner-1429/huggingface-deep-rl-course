# %%
import numpy as np
import gymnasium as gym
# import random
# import imageio
# import os
# import tqdm

# import pickle
from tqdm import tqdm

from unit2_hands_on.utils import initialize_q_table, train

# %%
env = gym.make("Taxi-v3", render_mode="rgb_array")

state_space = env.observation_space.n
action_space = env.action_space.n

# Create our Q table with state_size rows and action_size columns (500x6)
q_table_taxi = initialize_q_table(state_space, action_space)
print(q_table_taxi)
print("Q-table shape: ", q_table_taxi.shape)

# %%
# Training parameters
n_training_episodes = 25000  # Total training episodes
learning_rate = 0.7  # Learning rate

# Evaluation parameters
n_eval_episodes = 100  # Total number of test episodes

# DO NOT MODIFY EVAL_SEED
eval_seed = [
    16,
    54,
    165,
    177,
    191,
    191,
    120,
    80,
    149,
    178,
    48,
    38,
    6,
    125,
    174,
    73,
    50,
    172,
    100,
    148,
    146,
    6,
    25,
    40,
    68,
    148,
    49,
    167,
    9,
    97,
    164,
    176,
    61,
    7,
    54,
    55,
    161,
    131,
    184,
    51,
    170,
    12,
    120,
    113,
    95,
    126,
    51,
    98,
    36,
    135,
    54,
    82,
    45,
    95,
    89,
    59,
    95,
    124,
    9,
    113,
    58,
    85,
    51,
    134,
    121,
    169,
    105,
    21,
    30,
    11,
    50,
    65,
    12,
    43,
    82,
    145,
    152,
    97,
    106,
    55,
    31,
    85,
    38,
    112,
    102,
    168,
    123,
    97,
    21,
    83,
    158,
    26,
    80,
    63,
    5,
    81,
    32,
    11,
    28,
    148,
]  # Evaluation seed, this ensures that all classmates agents are trained on the same taxi starting position
# Each seed has a specific starting state

# Environment parameters
env_id = "Taxi-v3"  # Name of the environment
max_steps = 99  # Max steps per episode
gamma = 0.95  # Discounting rate

# Exploration parameters
max_epsilon = 1.0  # Exploration probability at start
min_epsilon = 0.05  # Minimum exploration probability
decay_rate = 0.005  # Exponential decay rate for exploration prob

# %%
# Train the Q-table
from unit2_hands_on.utils import train
q_table_taxi = initialize_q_table(state_space, action_space)
q_table_taxi = train(
    n_training_episodes,
    min_epsilon,
    max_epsilon,
    decay_rate,
    env,
    max_steps,
    q_table_taxi,
    gamma,
    learning_rate,
)
q_table_taxi

# %%
# Evaluation
model = {
    "env_id": env_id,
    "max_steps": max_steps,
    "n_training_episodes": n_training_episodes,
    "n_eval_episodes": n_eval_episodes,
    "eval_seed": eval_seed,
    "learning_rate": learning_rate,
    "gamma": gamma,
    "max_epsilon": max_epsilon,
    "min_epsilon": min_epsilon,
    "decay_rate": decay_rate,
    "qtable": q_table_taxi,
}

username = "user05181824"
repo_name = "q-Taxi-v3"

from unit2_hands_on.utils import push_to_hub
push_to_hub(repo_id=f"{username}/{repo_name}", model=model, env=env)
