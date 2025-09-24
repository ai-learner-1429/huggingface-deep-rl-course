# %%
import numpy as np
import gymnasium as gym
# import random
# import imageio
# import os
# import tqdm

# import pickle
from tqdm import tqdm

# %%
# Setup up env
env = gym.make(
    'FrozenLake-v1',
    desc=None,
    map_name="4x4",
    render_mode="rgb_array",
    is_slippery=False,
    # success_rate=1.0/3.0,
    # reward_schedule=(1, 0, 0)
)

# We create our environment with gym.make("<name_of_the_environment>")- `is_slippery=False`: The agent always moves in the intended direction due to the non-slippery nature of the frozen lake (deterministic).
print("_____OBSERVATION SPACE_____ \n")
print("Observation Space", env.observation_space)
print("Sample observation", env.observation_space.sample())  # Get a random observation

print("\n _____ACTION SPACE_____ \n")
print("Action Space Shape", env.action_space.n)
print("Action Space Sample", env.action_space.sample())  # Take a random action

# %%
# Initialize the Q-table
state_space = env.observation_space.n
action_space = env.action_space.n


def initialize_q_table(state_space: int, action_space: int):
    q_table = np.zeros((state_space, action_space))
    return q_table


q_table_frozenlake = initialize_q_table(state_space, action_space)

# %%
# Define the greedy and eps-greedy policy


def greedy_policy(q_table, state: int) -> int:
    # Exploitation: take the action with the highest state, action value
    action = int(np.argmax(q_table[state]))

    return action


def epsilon_greedy_policy(q_table, state, epsilon) -> int:
    # Randomly generate a number between 0 and 1
    random_num = np.random.rand()
    # if random_num > greater than epsilon --> exploitation
    if random_num > epsilon:
        action = greedy_policy(q_table, state)
    # else --> exploration
    else:
        action = np.random.randint(q_table.shape[1])

    return action


# %%
# Define the hyperparams

# Training parameters
n_training_episodes = 10000  # Total training episodes
learning_rate = 0.7  # Learning rate

# Evaluation parameters
n_eval_episodes = 100  # Total number of test episodes

# Environment parameters
env_id = "FrozenLake-v1"  # Name of the environment
max_steps = 99  # Max steps per episode
gamma = 0.95  # Discounting rate
eval_seed = []  # The evaluation seed of the environment

# Exploration parameters
max_epsilon = 1.0  # Exploration probability at start
min_epsilon = 0.05  # Minimum exploration probability
decay_rate = 0.0005  # Exponential decay rate for exploration prob

# %%
# Training loop

def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, q_table):

    for episode in range(n_training_episodes):
        eps = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        obs, _ = env.reset()
        step = 0
        terminated = False
        truncated = False
        while step < max_steps and not (terminated or truncated):
            action = epsilon_greedy_policy(q_table, obs, eps)
            new_obs, reward, terminated, truncated, _ = env.step(action)
            td_target = reward + gamma * np.max(q_table[new_obs])
            q_table[obs, action] += learning_rate * (td_target - q_table[obs, action])
            obs = new_obs

    return q_table


q_table_frozenlake = initialize_q_table(state_space, action_space)
q_table_frozenlake = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, q_table_frozenlake)

# %%
# Evaluation definition

def evaluate_agent(env, max_steps, n_eval_episodes, Q, seed):
    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    :param env: The evaluation environment
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param Q: The Q-table
    :param seed: The evaluation seed array (for taxi-v3)
    """
    episode_rewards = []
    for episode in tqdm(range(n_eval_episodes)):
        if seed:
            state, _ = env.reset(seed=seed[episode])
        else:
            state, _ = env.reset()
        total_rewards_ep = 0

        # Rollout for one episode
        for _ in range(max_steps):
            # Take the action (index) that have the maximum expected future reward given that state
            action = greedy_policy(Q, state)
            new_state, reward, terminated, truncated, _ = env.step(action)
            total_rewards_ep += reward

            if terminated or truncated:
                break
            state = new_state

        episode_rewards.append(total_rewards_ep)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward

# %%
# Evaluate our Agent
mean_reward, std_reward = evaluate_agent(env, max_steps, n_eval_episodes, q_table_frozenlake, eval_seed)
print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")