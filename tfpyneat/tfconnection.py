from __future__ import annotations
import random
from typing import List, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from .tfnode import TFNode


class TFConnection:
    """
    Represents the connection genes.
    A connection is a weighted, directed edge between two nodes (neurons).
    (Node) node_in: The incoming node (required)
    (Node) node_out: the outgoing node (required)
    (float) weight: Current weight of the connection
    """

    def __init__(self, node_in, node_out, inn_num, weight=None, enabled=True, mutation_likelihood=0.05):
        self.node_in: "TFNode" = node_in
        self.node_out: "TFNode" = node_out
        if weight:
            self.weight: float = weight
        else:
            self.weight: float = random.uniform(-1, 1)
        self.enabled: bool = enabled
        self.mutation_likelihood: float = mutation_likelihood
        self.inn_num = inn_num

    def mutate_weight(self) -> None:
        """
        'Connection weights mutate as in any NE system, with each
        connection either perturbed or not.'
        There are two ways that the weight can be perturbed: 
            1. Complete Perturbation - Randomly assigning a new weight within the range [-1,1]
            2. Gaussian Perturbation - Adjust the weight of the connection by a small gaussian value
        """
        if random.uniform(0, 1) < self.mutation_likelihood:
            self.weight = random.uniform(-1, 1)
        else:
            value = random.gauss(0, 1) / 50
            self.weight += value
            if self.weight > 1:
                self.weight = 1
            elif self.weight < -1:
                self.weight = -1

    def clone(self, from_node, to_node) -> "TFConnection":
        """
        Produces an identical clone of the connection.
        """
        clone = TFConnection(from_node,
                             to_node,
                             self.inn_num,
                             self.weight,
                             self.enabled,
                             self.mutation_likelihood
                             )

        return clone

    def get_innovation_num(self) -> int:
        """
        Returns the innovation number of the connection.
        """
        return self.inn_num

    def is_enabled(self) -> bool:
        return self.enabled

    def get_nodes(self) -> Tuple("Node", "Node"):
        """
        Returns a Tuple (incoming_node, outgoing_node).
        """
        return (self.node_in, self.node_out)
