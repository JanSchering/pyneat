from __future__ import annotations
from typing import Any, Callable, List, Tuple
import pickle
import time
import os
import csv
import gym
import pybullet_envs
import pybulletgym
from interface_neat import clamp


def run_game(genome):
    env = gym.make("InvertedPendulumMuJoCoEnv-v0")
    # Ensure that the environment is displayed
    env.render(mode="human")
    # Get the initial state of the environment
    observation = env.reset()
    # Track the total score achieved
    total_reward = 0
    # Run the game infinitely
    while True:
        # Forward pass the state through the network
        outputs = genome.forward(observation)
        # Move the game forward
        observation, reward, done, info = env.step(clamp(outputs))
        # add on the reward
        total_reward += reward
        if done:
            print(f"GAME OVER - FINAL SCORE: {total_reward}")
            # reset the game
            total_reward = 0
            env.reset()
            print(f"STARTING NEW ROUND")


if __name__ == "__main__":
    # Path to the stored Genome
    genome_path = os.path.join(os.getcwd(), "winner.pickle")
    # Load the Genome from file
    with open('config.dictionary', 'rb') as config_dictionary_file:
        genome = pickle.load(genome_path)
