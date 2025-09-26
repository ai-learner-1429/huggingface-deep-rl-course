# https://huggingface.co/learn/deep-rl-course/en/unit6/hands-on

# %%
# import os

import gymnasium as gym
# import panda_gym

from huggingface_sb3 import load_from_hub, package_to_hub

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

# from huggingface_hub import notebook_login

# %%
env_id = "PandaReachDense-v3"

# Create the env
env = gym.make(env_id)

# Get the state space and action space
obs_space = env.observation_space
act_space = env.action_space

print("_____OBSERVATION SPACE_____ \n")
print("The State Space is: ", obs_space)
print("Sample observation", env.observation_space.sample()) # Get a random observation

print("\n _____ACTION SPACE_____ \n")
print("The Action Space is: ", act_space)
print("Action Space Sample", env.action_space.sample()) # Take a random action

# %%
env = make_vec_env(env_id, n_envs=4)

# Adding this wrapper to normalize the observation and the reward
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

# %%
# Train and save the model
model = A2C(policy="MultiInputPolicy", env=env, verbose=1)

# Takes 18min to train 1e6 steps, mean reward = -0.27 +/- 0.10
model.learn(1_000_000)

# Save the model and  VecNormalize statistics when saving the agent
model.save("a2c-PandaReachDense-v3")
env.save("vec_normalize.pkl")

# %%
# Evaluate the agent

# Load the saved statistics
eval_env = DummyVecEnv([lambda: gym.make("PandaReachDense-v3")])
eval_env = VecNormalize.load("vec_normalize.pkl", eval_env)

# We need to override the render_mode
eval_env.render_mode = "rgb_array"

#  do not update them at test time
eval_env.training = False
# reward normalization is not needed at test time
eval_env.norm_reward = False

# Load the agent
model = A2C.load("a2c-PandaReachDense-v3")

mean_reward, std_reward = evaluate_policy(model, eval_env)

print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")

# %%
# Publish model to Hugging Face

from huggingface_sb3 import package_to_hub

package_to_hub(
    model=model,
    model_name=f"a2c-{env_id}",
    model_architecture="A2C",
    env_id=env_id,
    eval_env=eval_env,
    repo_id=f"user05181824/a2c-{env_id}", # Change the username
    commit_message="Initial commit",
)