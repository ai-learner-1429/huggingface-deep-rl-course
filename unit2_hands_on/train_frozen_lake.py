# %%
import gymnasium as gym

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

from unit2_hands_on.utils import initialize_q_table
q_table_frozenlake = initialize_q_table(state_space, action_space)

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

from unit2_hands_on.utils import train

q_table_frozenlake = initialize_q_table(state_space, action_space)
q_table_frozenlake = train(
    n_training_episodes,
    min_epsilon,
    max_epsilon,
    decay_rate,
    env,
    max_steps,
    q_table_frozenlake,
    gamma,
    learning_rate,
)

# %%
# Evaluate our Agent
from unit2_hands_on.utils import evaluate_agent
mean_reward, std_reward = evaluate_agent(env, max_steps, n_eval_episodes, q_table_frozenlake, eval_seed)
print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

# %%
# Upload the model to Hugging Face


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
    "qtable": q_table_frozenlake,
}

username = "user05181824"
repo_name = "q-FrozenLake-v1-4x4-noSlippery"

from unit2_hands_on.utils import push_to_hub
push_to_hub(repo_id=f"{username}/{repo_name}", model=model, env=env)