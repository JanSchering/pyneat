from __future__ import annotations
from typing import Any, Callable, List, Tuple
import pickle
import time
import os
import csv
import gym
import pybullet_envs
import pybulletgym
from interface_tfneat import softmax, enjoy
import numpy as np


if __name__ == "__main__":
    # Path to the stored Genome
    genome_path = os.path.join(os.getcwd(), "winner.pickle")
    # Load the Genome from file
    with open(genome_path, 'rb') as file:
        genome = pickle.load(file)

    enjoy(genome)
