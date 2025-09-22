import gymnasium as gym

env = gym.make("LunarLander-v2", render_mode="human")  # render_mode="human" needed for visualization
obs = env.reset()

done = False
while not done:
    action = env.action_space.sample()  # replace with your policy
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()