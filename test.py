import gym
env = gym.make('MountainCar-v0')
env.reset()
print(f"OBS: {env.observation_space} ACT: {env.action_space}")
for _ in range(1000):
    _, reward, _, _ = env.step(env.action_space.sample())
    print(reward)
    env.render('human')
env.close()
