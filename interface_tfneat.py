# %%
from __future__ import annotations
from typing import Any, Callable, List, Tuple
from tfpyneat import TFGenome
from tfpyneat import TFPopulation
import gym
import numpy as np
import tfpyneat.innovationcounter as innovationcounter
import pickle
import time
import os
import csv

# utils for the learning process
runs_per_net = 5
max_generations = 200
threshold = 200

# stat utils
CSV_HEADER = ["generation, species, highscore"]
STAT_PATH = os.path.join(os.getcwd(), "stats_tfpyneat")

env = gym.make("LunarLander-v2")
env.seed(0)


def enjoy(genome):
    global env
    # Get the initial state of the environment
    observation = env.reset()
    # Track the total score achieved
    total_reward = 0
    # Run the game infinitely
    while True:
        # Forward pass the state through the network
        outputs = softmax(genome.forward(observation))
        # Move the game forward
        observation, reward, done, info = env.step(np.argmax(outputs))
        # add on the reward
        total_reward += reward
        if done:
            print(f"GAME OVER - FINAL SCORE: {total_reward}")
            # reset the game
            total_reward = 0
            env.reset()
            print(f"STARTING NEW ROUND")
        else:
            env.render(mode="human")


def clamp(l: List[float]) -> List[float]:
    """
    Custom clamping activation function for the Genomes' output
    """
    return [1 if x > 1 else -1 if x < -1 else x for x in l]


def softmax(x: List[float]) -> List[float]:
    """
    Apply softmax on a list of values
    """
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x))
    return f_x


def eval_genome(genome: TFGenome) -> float:
    """
    The fitness criterion that will be used to evaluate the Genomes 
    in the natural selection process
    """
    fitnesses = []

    for runs in range(runs_per_net):
        global env
        # env.render(mode=None)
        observation = env.reset()
        fitness = 0.0
        done = False
        while not done:
            outputs = softmax(genome.forward(observation))
            # Move the game forward
            observation, reward, done, info = env.step(np.argmax(outputs))
            fitness += reward

        fitnesses.append(fitness)

    return np.mean(fitnesses)


if __name__ == '__main__':
    for blubbb in range(1):
        # The innovation counter is needed to track the novel mutations in the Population
        innovationcounter.init()
        # Initialize the population of genomes
        population = TFPopulation(8, 4, eval_genome, 100)
        # Keep track of whether the learning process was successful
        found_winner = False
        # Keep track of the all-time highscore achieved during training
        highest_performance = 0
        # Create a file to track the statistics of the learning process
        timestamp = time.time()
        with open(os.path.join(STAT_PATH, f"run_{timestamp}.csv"), "w") as stat_file:
            # initialize a file writer
            stat_writer = csv.writer(stat_file)
            for i in range(max_generations):
                # Move the population a generation forward by performing natural selection
                population.natural_selection()
                # Retrieve the highscore of the population for the current generation
                gen_highscore = population.best_score
                # Track the current highest all-time performance
                if gen_highscore > highest_performance:
                    highest_performance = gen_highscore
                # Print out some stats for the user
                print(f"Generation: {i}")
                print(f"Best Score: {gen_highscore}")
                print(f"Species: {len(population.species)}")
                # Update Statistics
                stat_writer.writerow(
                    [i, len(population.species), gen_highscore])
                # Check if the learning process is finished
                if population.best_genome.unadjusted_fitness > threshold:
                    print("Found a winner!")
                    found_winner = True
                    # Save the winning genome
                    with open(f"winner.pickle", "wb") as file:
                        pickle.dump(population.best_genome, file)
                        enjoy(population.best_genome)
                    break
            # Learning Process ended without finding a strong enough Genome
            if not found_winner:
                print(
                    f"No Genome was able to match the Threshold, highest Performance: {highest_performance}")

# %%
