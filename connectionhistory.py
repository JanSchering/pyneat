from __future__ import annotations
import random
from typing import List, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from node import Node
    from genome import Genome


class ConnectionHistory:
    """
    Represents the History of the Genome mutations
    """

    def __init__(self, from_node: int, to_node: int, inn_num: int, numbers: List[int]):
        self.from_node_num = from_node
        self.to_node_num = to_node
        self.inn_num = inn_num

        self.inn_numbers = [n for n in numbers]

    def matches(self, genome: Genome, from_node: Node, to_node: Node):
        """
        Checks if genome matches the original genome and the nodes are matching
        """
        # Checks if the number of connections are different
        if len(genome.connections) == len(self.inn_numbers):
            # Make sure that the numbers of the nodes are matching
            if from_node.num == self.from_node_num and to_node.num == self.to_node_num:
                # For every connection in the Genome, check if it is tracked in the history
                for conn in genome.connections:
                    if not conn.inn_num in self.inn_numbers:
                        return False
                # Every check was passed
                return True
        else:
            return False
