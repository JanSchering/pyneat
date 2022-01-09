import gym
import pybulletgym
env = gym.make('InvertedPendulumMuJoCoEnv-v0')
env.reset()
print(f"OBS: {env.observation_space} ACT: {env.action_space}")
for _ in range(1000):
    _, reward, done, _ = env.step(env.action_space.sample())
    env.render('human')
env.close()
