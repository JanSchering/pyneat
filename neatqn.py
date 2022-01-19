from __future__ import annotations
import gym
import random
import tensorflow as tf
import numpy as np
import time
from collections import deque
from tfpyneat import TFPopulation, TFGenome
import tfpyneat.innovationcounter as innovationcounter
from typing import Any, Callable, List, Tuple
import params
import tensorflow as tf
from dqn import DQNAgent, run_dqn


def eval_genome(genome: TFGenome) -> float:
    """
    The fitness criterion that will be used to evaluate the Genomes 
    in the natural selection process
    """
    fitnesses = []
    for runs in range(params.RUNS_PER_NET):
        global env
        # env.render(mode=None)
        observation = env.reset()
        fitness = 0.0
        done = False
        while not done:
            outputs = genome.forward(observation)
            # Move the game forward
            observation, reward, done, info = env.step(np.argmax(outputs))
            fitness += reward
        fitnesses.append(fitness)
    return np.min(fitnesses)


env = gym.make("LunarLander-v2")
env.seed(0)
random.seed(0)
tf.random.set_seed(0)
innovationcounter.init()

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

population = TFPopulation(obs_dim, act_dim, eval_genome, 100)

# Run a number of generations of natural selection to develop a network structure
for gen in range(params.MAX_GENERATIONS):
    population.natural_selection()
    # print out some statistics
    layer_info = set()
    for g in population.genomes:
        layer_info.add(g.layers)
    score = population.best_genome.unadjusted_fitness
    num_nodes = len(population.best_genome.nodes)
    best_layers = population.best_genome.layers
    num_species = len(population.species)
    print(f"GEN: {gen+1} SCORE: {score} SPECIES: {num_species} LAYERS: {layer_info}  L: {best_layers} NODES: {num_nodes}")

main_model = population.best_genome.create_model()
target_model = population.best_genome.create_model()
agent = DQNAgent(obs_dim, act_dim, models=(main_model, target_model))

run_dqn(agent, env)
