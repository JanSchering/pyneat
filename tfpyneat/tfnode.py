from __future__ import annotations
from enum import Enum
import random
from typing import List, Tuple
from ..pyneat.connection import Connection
import math
import keras


class TFNode:
    """
    Represents a Node (Neuron) gene.
    (int) num: The innovation number of the gene. Used as historical marker
    (int) layer: The layer that the Node is in
    (boolean) out: Whether the node is an output node
    """

    def __init__(self, num, layer=None, is_out=False, activation="sigmoid"):
        self.num: int = num
        self.input_val = 0
        self.out_val = 0
        self.out_conn: List[Connection] = []
        self.layer: int = layer

        self.is_out: bool = is_out

        if activation == "sigmoid":
            self.activation = keras.activations.sigmoid
        elif activation == "tanh":
            self.activation = keras.activations.tanh
        elif activation == "relu":
            self.activation = keras.activations.relu

    def mutate_activation(self) -> None:
        """
        Randomly assigns a new activation function to the node 
        """
        self.activation = random.choice(list(
            [keras.activations.relu, keras.activations.tanh, keras.activations.sigmoid]))

    def clone(self) -> "TFNode":
        cloned = TFNode(self.num, self.layer, self.is_out)
        cloned.activation = self.activation  # Ensure same activation
        return cloned

    def get_layer(self) -> int:
        return self.layer

    def get_outgoing_connections(self) -> List[Connection]:
        """
        returns a list of connections that have this Node as their outgoing node.
        """
        return self.out_conn

    def has_connection(self, node: "TFNode"):
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
