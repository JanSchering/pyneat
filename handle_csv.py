# %%
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import gym

data_path = os.path.join(os.getcwd(), "data")
# %%
# Collect data for a baseline random agent
env = gym.make("LunarLander-v2")
env.seed(1)

mean_rewards = []
max_rewards = []
min_rewards = []
episode_rewards = deque(maxlen=100)
# As we log the initial performance of the agent during training,
# There is a total of 21 entries that we are matching for the 2000 episodes
for i in range(2001):
    ep_rew = 0
    done = False
    env.reset()
    # Run an episode sampling random actions
    while not done:
        obs, r, done, _ = env.step(env.action_space.sample())
        ep_rew += r
    # Collect the reward of the episode
    episode_rewards.append(ep_rew)
    # every 100 episodes, collect a new data point
    if i % 100 == 0:
        mean_rewards.append(np.mean(episode_rewards))
        max_rewards.append(np.max(episode_rewards))
        min_rewards.append(np.min(episode_rewards))


# %%
"""
Create plot for maximum cumulative performance comparison
"""
data = pd.read_csv(os.path.join(data_path, "reward_max.csv"))
plt.plot(data["Step"][:21], data["Value"][:21])
data = pd.read_csv(os.path.join(data_path, "nopos_rmax.csv"))
plt.plot(data["Step"], data["Value"])
data = pd.read_csv(os.path.join(data_path, "novel_max.csv"))
plt.plot(data["Step"], data["Value"])
plt.plot(data["Step"][:21], max_rewards)
plt.xlabel("episode")
plt.ylabel("reward")
plt.legend(['Fully Observable', 'No Position',
           "No Velocity", "Baseline Random"])
plt.title("Maximum cumulative reward per 100 episodes of training")
plt.savefig('foo.jpg')
plt.show()

# %%
"""
Create Plot for mean cumulative performance comparison
"""
data = pd.read_csv(os.path.join(data_path, "reward_mean.csv"))
plt.plot(data["Step"][:21], data["Value"][:21])
data = pd.read_csv(os.path.join(data_path, "nopos_rmean.csv"))
plt.plot(data["Step"], data["Value"])
data = pd.read_csv(os.path.join(data_path, "novel_mean.csv"))
plt.plot(data["Step"], data["Value"])
plt.plot(data["Step"][:21], mean_rewards)
plt.xlabel("episode")
plt.ylabel("reward")
plt.legend(['Fully Observable', 'No Position',
           "No Velocity", "Baseline Random"])
plt.title("Mean cumulative reward averaged over 100 episodes of training")
plt.savefig('foo.jpg')
plt.show()

# %%
"""
Create Plot for the minimum cumulative reward comparison
"""
data = pd.read_csv(os.path.join(data_path, "full_min.csv"))
plt.plot(data["Step"][:21], data["Value"][:21])
data = pd.read_csv(os.path.join(data_path, "nopos_min.csv"))
plt.plot(data["Step"], data["Value"])
data = pd.read_csv(os.path.join(data_path, "novel_min.csv"))
plt.plot(data["Step"], data["Value"])
plt.plot(data["Step"][:21], min_rewards)
plt.xlabel("episode")
plt.ylabel("reward")
plt.legend(['Fully Observable', 'No Position',
           "No Velocity", "Baseline Random"])
plt.title("Minimum cumulative reward averaged over 100 episodes of training")
plt.savefig('foo.jpg')
plt.show()
