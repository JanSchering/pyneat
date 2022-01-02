# %%
from __future__ import annotations
from typing import Any, Callable, List, Tuple
from population import Population
import gym
import pybullet_envs
import pybulletgym
import numpy as np
import innovationcounter
import pickle

runs_per_net = 2
max_generations = 200
threshold = 950


def create_env():
    env = gym.make("InvertedPendulumMuJoCoEnv-v0")
    return env


def clamp(l):
    return [1 if x > 1 else -1 if x < -1 else x for x in l]


def eval_genome(genome):

    fitnesses = []

    for runs in range(runs_per_net):
        env = create_env()
        # env.render(mode=None)
        observation = env.reset()
        fitness = 0.0
        done = False
        while not done:
            outputs = genome.forward(observation)
            # Action is the highest value of the outputs
            #action = sorted(outputs, key=lambda x: x)[-1]
            # Move the game forward
            observation, reward, done, info = env.step(clamp(outputs))
            fitness += reward

        fitnesses.append(fitness)

    return np.mean(fitnesses)


if __name__ == '__main__':
    innovationcounter.init()
    population = Population(4, 1, create_env, eval_genome, 200)
    found_winner = False
    highest_performance = 0
    for i in range(max_generations):
        population.natural_selection()
        gen_highscore = population.best_score
        if gen_highscore > highest_performance:
            highest_performance = gen_highscore
        print(f"Generation: {i}")
        print(f"Best Score: {gen_highscore}")
        print(f"Species: {len(population.species)}")
        if population.best_genome.unadjusted_fitness > threshold:
            print("Found a winner!")
            found_winner = True
            with open("winner.pickle", "wb") as file:
                pickle.dump(population.best_genome, file)
            break

    if not found_winner:
        print(
            f"No Genome was able to match the Threshold, highest Performance: {highest_performance}")
