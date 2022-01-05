from __future__ import annotations
from enum import Enum
import random
from typing import List, Tuple
from .connection import Connection
import math


class Activation(Enum):
    SIGMOID = "sigmoid"
    TANH = "tanh"
    RELU = "relu"
    IDENTITY = "identity"
    STEP = "step"
    CLAMPED = "clamped"


class Node:
    """
    Represents a Node (Neuron) gene.
    (int) num: The innovation number of the gene. Used as historical marker
    (int) layer: The layer that the Node is in
    (boolean) out: Whether the node is an output node
    """

    def __init__(self, num, layer=None, is_out=False):
        self.num: int = num
        self.input_val = 0
        self.out_val = 0
        self.out_conn: List[Connection] = []
        self.layer: int = layer

        self.is_out: bool = is_out
        self.activation = Activation.CLAMPED
        #self.activation: Activation = random.choice(list(Activation))

    def activate_node(self) -> None:
        """
        Calculates the activation of the node and publishes the result to all registered connections
        """
        # We only need to activate nodes that are not inputs
        if self.layer != 0 and not self.is_out:
            self.out_val = self.apply_activation(self.input_val)

        if self.is_out:
            self.out_val = self.input_val

        # Publish the node output to the connections.
        for conn in self.out_conn:
            if conn.is_enabled():  # Only publish for active connections
                conn.node_out.input_val += conn.weight * self.out_val

    def mutate_activation(self) -> None:
        """
        Randomly assigns a new activation function to the node 
        """
        pass
        #self.activation = random.choice(list(Activation))

    def clone(self) -> "Node":
        cloned = Node(self.num, self.layer, self.is_out)
        cloned.activation = self.activation  # Ensure same activation
        return cloned

    def get_layer(self) -> int:
        return self.layer

    def get_outgoing_connections(self) -> List[Connection]:
        """
        returns a list of connections that have this Node as their outgoing node.
        """
        return self.out_conn

    def has_connection(self, node: "Node"):
        """
        Checks if there is a registered connection between the two nodes
        (Node) node: The node to check
        """
        # nodes on the same layer have no connections
        if node.get_layer() == self.layer:
            return False
        # If node is in lower layer, check if there is a connection from it to this node
        elif node.get_layer() < self.layer:
            for conn in node.get_connections():
                if conn.get_nodes()[1] == self:
                    return True
        # Else check if there is a connection from this node to the other one
        else:
            for conn in self.out_conn:
                if conn.get_nodes()[1] == node:
                    return True
        # No matching connection found
        return False

    def apply_activation(self, x) -> float:
        """
        Helper-Function: Applies the current activation function of the node to a given value.
        (int) x: The value to apply the function to.
        """
        if self.activation == Activation.IDENTITY:
            return x
        elif self.activation == Activation.SIGMOID:
            return 1 / (1 + math.exp(-x))
        elif self.activation == Activation.RELU:
            return max(0.0, x)
        elif self.activation == Activation.TANH:
            return math.tanh(x)
        elif self.activation == Activation.STEP:
            return 1 if x > 0 else 0
        elif self.activation == Activation.CLAMPED:
            return 1 if x > 1 else -1 if x < -1 else x
