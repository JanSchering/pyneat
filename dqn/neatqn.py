from __future__ import annotations
from typing import Any, Callable, List, Tuple
from tfpyneat import TFPopulation, TFGenome
from ..interface_tfneat import eval_genome
from collections import deque
import params


class NEATQAgent:
    """
    Implements a combination of NEAT and Deep Q Learning.
    """

    def __init__(self, obs_dim: int, act_dim: int):
        self.population = TFPopulation(
            obs_dim, act_dim, eval_genome, 50)
        self.replay_memory = deque(params.MAX_MEMORY_SIZE)
