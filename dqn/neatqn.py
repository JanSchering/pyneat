from __future__ import annotations
from typing import Any, Callable, List, Tuple
from tfpyneat import TFPopulation, TFGenome
from collections import deque
from .dqn import MAX_MEMORY_SIZE


class NEATQAgent:
    """
    Implements a combination of NEAT and Deep Q Learning.
    """

    def __init__(self, obs_dim: int, act_dim: int):
        self.population = TFPopulation(
            obs_dim, act_dim, self.run_generation, 50)
        self.replay_memory = deque(MAX_MEMORY_SIZE)

    def run_generation(self, genome: TFGenome):
        pass
