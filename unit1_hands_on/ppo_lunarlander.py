# Tutorial: https://huggingface.co/learn/deep-rl-course/en/unit1/hands-on

# %%
import gymnasium as gym

from huggingface_sb3 import load_from_hub, package_to_hub
from huggingface_hub import (
    notebook_login,
)  # To log to our Hugging Face account to be able to upload models to the Hub.

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

# %%
# Test a random policy.

# First, we create our environment called LunarLander-v2
env = gym.make("LunarLander-v2")
env_id = env.spec.id

# Then we reset this environment
observation, info = env.reset()

for _ in range(20):
    # Take a random action
    action = env.action_space.sample()
    print("Action taken:", action)

    # Do this action in the environment and get
    # next_state, reward, terminated, truncated and info
    observation, reward, terminated, truncated, info = env.step(action)

    # If the game is terminated (in our case we land, crashed) or truncated (timeout)
    if terminated or truncated:
        # Reset the environment
        print("Environment is reset")
        observation, info = env.reset()

env.close()

# %%
# Inspect observation_space and action_space.

# We create our environment with gym.make("<name_of_the_environment>")
env = gym.make(env_id)
env.reset()
print("_____OBSERVATION SPACE_____ \n")
print("Observation Space Shape", env.observation_space.shape)
print("Sample observation", env.observation_space.sample())  # Get a random observation

print("\n _____ACTION SPACE_____ \n")
print("Action Space Shape", env.action_space.n)
print("Action Space Sample", env.action_space.sample())  # Take a random action

# %%
# Create environment
# env = gym.make(env_id)
n_envs = 16
env = make_vec_env(env_id, n_envs=n_envs)
# env = make_vec_env(env_id, n_envs=n_envs, vec_env_cls=SubprocVecEnv)

# SOLUTION
# We added some parameters to accelerate the training
model = PPO(
    policy="MlpPolicy",
    env=env,
    # Update params
    n_steps=1024,
    batch_size=64,
    n_epochs=4,
    # PPO params
    gamma=0.999,
    gae_lambda=0.98,
    ent_coef=0.01,
    # Other
    device="auto",
    verbose=1,
)

# SOLUTION
# Train it for 1,000,000 timesteps
# v1 training stats: fps=694, 1.47s per iteration (1024 steps), 1440s for 977 iterations or 1e6 steps. Result: 250.22 +/- 32.58
# v2, n_envs 1 -> 16, reduces runtime from 1440s to 588s, a 2.4x speedup. Result: 277.43 +/- 15.65
model.learn(total_timesteps=int(1e6))

# Save the model inzip a zip file.
model_name = f"ppo-{env_id}"
model.save(model_name)

# %%
# Evaluate the model
eval_env = Monitor(gym.make(env_id))
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

# %%
# Upload the model to HuggingFace.
# First, call "hf auth login" on the command line with a HF token.

from stable_baselines3.common.vec_env import DummyVecEnv

from huggingface_sb3 import package_to_hub

# repo_id is the id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name} for instance ThomasSimonini/ppo-LunarLander-v2
repo_id = "user05181824/ppo-LunarLander-v2"

# Create the evaluation env and set the render_mode="rgb_array"
eval_env = DummyVecEnv([lambda: gym.make(env_id, render_mode="rgb_array")])

model_architecture = model.__class__.__name__

commit_message = "Upload PPO LunarLander-v2 trained agent"

# method save, evaluate, generate a model card and record a replay video of your agent before pushing the repo to the hub
package_to_hub(
    model=model,  # Our trained model
    model_name=model_name,  # The name of our trained model
    model_architecture=model_architecture,  # The model architecture we used: in our case PPO
    env_id=env_id,  # Name of the environment
    eval_env=eval_env,  # Evaluation Environment
    repo_id=repo_id,  # id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name} for instance ThomasSimonini/ppo-LunarLander-v2
    commit_message=commit_message,
)