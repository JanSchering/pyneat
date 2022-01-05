from __future__ import annotations
from typing import Any, Callable, List, Tuple
from ..pyneat import Population, Genome
from collections import deque

MAX_REPLAY_SIZE = 10_000


class NEATQAgent:
    """
    Implements a combination of NEAT and Deep Q Learning.
    """

    def __init__(self, obs_dim: int, act_dim: int):
        self.population = Population(obs_dim, act_dim, self.run_generation, 50)
        self.replay_memory = deque(MAX_REPLAY_SIZE)

    def run_generation(self, genome: Genome):
        pass

    def backprop(self):
        pass
