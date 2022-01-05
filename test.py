import gym
env = gym.make('Breakout-ram-v0')
env.reset()
print(f"OBS: {env.observation_space} ACT: {env.action_space}")
for _ in range(1000):
    env.step(env.action_space.sample())
    env.render('human')
env.close()
