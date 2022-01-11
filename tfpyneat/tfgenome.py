from __future__ import annotations
from typing import Callable, List

from keras import layers
from .tfnode import TFNode
from .tfconnection import TFConnection
from .connectionhistory import ConnectionHistory
from . import innovationcounter
import random
import math
import keras
import tensorflow as tf
import numpy as np


class TFGenome:
    """
    Represents a network topology through a list of Connections.
    """

    def __init__(self, num_in, num_out, crossover):
        self.num_in: int = num_in
        self.num_out: int = num_out

        self.layers: int = 2

        # Attributes used to track performance of the Genome during evolution
        self.fitness: float = 0
        self.unadjusted_fitness: float = 0
        self.best_score = 0
        self.score = 0
        self.lifespan = 0
        self.done = False

        self.id_count: int = 0  # Tracks the number of created nodes

        self.nodes: List[TFNode] = []
        self.connections: List[TFConnection] = []
        self.network: List[TFNode] = []

        if crossover:
            return

        # Create and collect a Node for each input
        for i in range(self.num_in):
            node = TFNode(self.id_count, layer=0)
            self.id_count += 1
            self.nodes.append(node)

        # Create and collect a Node for each output
        for i in range(self.num_out):
            node = TFNode(self.id_count, layer=1, is_out=True)
            self.id_count += 1
            self.nodes.append(node)

    def create_model(self):
        """
        turns the nodes into a Tensorflow model
        """
        self.sort_nodes_by_layer()
        outputs = {}
        inputs = []

        for layer in range(self.layers):
            nodes = [n for n in self.nodes if n.layer == layer]

            if layer == 0:
                for node in nodes:
                    outputs[node.num] = keras.Input(shape=(1,))
                    inputs.append(outputs[node.num])
                    #outputs[node.num] = node.tfnode
            else:
                for node in nodes:
                    conns = [
                        c for c in self.connections if c.node_out.num == node.num]
                    node_ins = []
                    weights = []
                    for conn in conns:
                        layer_val = outputs[conn.node_in.num]
                        node_ins.append(layer_val)
                        weights.append(conn.weight)

                    if len(node_ins) == 1:
                        node_input = node_ins[0]
                    else:
                        node_input = keras.layers.Concatenate()(node_ins)
                    if not node.is_out:
                        outputs[node.num] = keras.layers.Dense(
                            1,
                            activation=node.get_tfactivation(),
                            kernel_initializer=self.init_weight(weights),
                            bias_initializer=self.init_bias(node)
                        )(node_input)
                    else:
                        outputs[node.num] = keras.layers.Dense(
                            1,
                            kernel_initializer=self.init_weight(weights),
                            bias_initializer=self.init_bias(node)
                        )(node_input)

        output_nodes = [outputs[n.num] for n in self.nodes if n.is_out]
        if len(output_nodes) == 1:
            out = output_nodes[0]
        else:
            out = keras.layers.Concatenate()(output_nodes)
        self.model = keras.Model(inputs=inputs, outputs=out)

    def fully_connect(self, innovationhistory):
        """
        Adds Connections to the Genome to create a fully connected phenotype
        """
        for i in range(self.num_in):
            for j in range(self.num_in, len(self.nodes)):
                node_in = self.nodes[i]
                node_out = self.nodes[j]
                conn_num = self.get_inn_num(
                    innovationhistory, node_in, node_out)
                self.connections.append(
                    TFConnection(node_in, node_out, conn_num))

    def init_bias(self, node: TFNode):
        """
        Creates a bias initializer function for the tf model
        """
        bias = np.array([node.bias], dtype="float32")
        bias_tensor = tf.Variable(tf.convert_to_tensor(bias))

        def initializer(shape, dtype):
            return bias_tensor
        return initializer

    def init_weight(self, weights):
        """
        Creates a weight initializer function for the tf model
        """
        weight_batch = np.array(weights, dtype="float32")[:, np.newaxis]
        weight_tensor = tf.Variable(tf.convert_to_tensor(weight_batch))

        def initializer(shape, dtype):
            return weight_tensor
        return initializer

    def connect_nodes(self):
        """
        Adds outgoing connections of a node to that node so that it can acess the next node during forward pass
        """
        for node in self.nodes:
            node.out_conn.clear()

        for conn in self.connections:
            conn.node_in.out_conn.append(conn)

    def forward(self, inputs) -> List[float]:
        """
        Calculate forward pass through the net
        """
        self.sort_nodes_by_layer()
        self.connect_nodes()

        # Clear the old inputs of each node
        for node in self.nodes:
            node.input_val = 0
            node.out_val = 0

        # Publish the input values to the input nodes
        for idx, input in enumerate(inputs):
            self.nodes[idx].out_val = input

        result = []

        for node in self.nodes:
            # Activate each node
            node.activate_node()
            # If the node is an output node, collect the result from it
            if node.is_out:
                result.append(node.out_val)

        return result

    def produce_offspring(self, partner: "TFGenome") -> "TFGenome":
        """
        Crosses over itself with another Genome to produce an offspring
        """
        offspring = TFGenome(self.num_in, self.num_out, True)
        # Make sure they are using the same innovation count
        offspring.id_count = self.id_count
        offspring.connections = []
        offspring.nodes = []
        offspring.layers = self.layers

        child_connections: List[TFConnection] = []
        is_enabled = []

        for conn in self.connections:
            set_enabled = True

            idx = partner.find_conn_idx(conn)
            if idx != -1:
                partner_conn = partner.connections[idx]
                if not(conn.is_enabled()) or not(partner_conn.is_enabled):
                    if random.uniform(0, 1) < 0.75:
                        set_enabled = False

                if random.uniform(0, 1) < 0.5:
                    child_connections.append(conn)
                else:
                    child_connections.append(partner_conn)
            else:
                child_connections.append(conn)
                set_enabled = conn.is_enabled()

            is_enabled.append(set_enabled)

        # Create a copy of all nodes and assign them to the offspring
        # If output node, there is a 50 % chance that the activation will be
        # the same as this Genome's activation, else it will be the partner's
        # activation
        for node in self.nodes:
            clone = node.clone()
            if clone.is_out and random.randint(0, 1) == 1:
                # retrieve the correct node from the partner
                partner_node = partner.get_node(clone.num)
                clone.bias = partner_node.bias
                clone.activation = partner_node.activation
            offspring.nodes.append(clone)

        for idx, child_conn in enumerate(child_connections):
            in_n, out_n = child_conn.get_nodes()
            cnode_in = offspring.get_node(in_n.num)
            cnode_out = offspring.get_node(out_n.num)
            conn_clone = child_conn.clone(cnode_in, cnode_out)
            conn_clone.enabled = is_enabled[idx]
            offspring.connections.append(conn_clone)

        offspring.connect_nodes()
        return offspring

    def mutate_genome(self, innovationhistory) -> None:
        """
        Handles the Possible Mutations of the Genome
        """
        # 80% chance to mutate the weight of the connections
        if random.uniform(0, 1) < 0.8:
            for conn in self.connections:
                conn.mutate_weight()

        if random.uniform(0, 1) < 0.5:
            for node in self.nodes:
                node.mutate_bias()

        # 10% chance to mutate the activation functions
        if random.uniform(0, 1) < 0.1:
            for node in self.nodes:
                node.mutate_activation()

        # 5% chance that the genome mutates and adds a new connection
        if random.uniform(0, 1) < 0.05:
            self.add_connection(innovationhistory)

        # 1% chance that the genome mutates and adds a new Node
        if random.uniform(0, 1) < 0.1:
            self.add_node(innovationhistory)

    def add_node(self, innovationhistory) -> None:
        """
        Executes the Mutation of a Genome in which an existing connection is split
        and a new Node is added
        """
        if len(self.connections) == 0:
            self.add_connection(innovationhistory)

        # Get connection to split
        conn = self.connections[random.randint(0, len(self.connections)-1)]

        # Disable the original connection
        conn.enabled = False

        # Stop tracking the old Connection
        self.connections.remove(conn)

        # Create the new Node
        newnode_num = self.id_count
        new_node = TFNode(newnode_num)
        self.id_count += 1

        # Create the new Connections for the node
        inn_num = self.get_inn_num(innovationhistory, new_node, conn.node_out)
        con_out = TFConnection(new_node, conn.node_out,
                               inn_num, conn.weight)
        inn_num = self.get_inn_num(innovationhistory, conn.node_in, new_node)
        con_in = TFConnection(conn.node_in, new_node,
                              inn_num, weight=1)

        self.connections.append(con_in)
        self.connections.append(con_out)

        new_node.layer = conn.node_in.layer+1

        # Check if the connection was a direct connection or a long connection
        # If direct connection, we added a new layer and every higher layer has to be incremented
        if new_node.layer == conn.node_out.layer:
            for node in self.nodes:
                if node.layer >= new_node.layer:
                    node.layer += 1

            self.layers += 1

        # Track new node
        self.nodes.append(new_node)
        self.connect_nodes()

    def add_connection(self, innovationhistory) -> None:
        """
        Executes the Mutation in which a new connection between two random Nodes is
        established
        """
        # Check if the Network is already fully connected
        if self.is_fully_connected():
            return

        # Search for a valid combination of nodes
        while True:
            node1 = self.nodes[random.randint(0, len(self.nodes)-1)]
            node2 = self.nodes[random.randint(0, len(self.nodes)-1)]

            if node1.layer != node2.layer and not self.is_connected(node1, node2):
                break

        # Create a forward Connection
        weight = random.uniform(-1, 1)
        if node1.layer > node2.layer:
            inn_num = self.get_inn_num(innovationhistory, node2, node1)
            new_conn = TFConnection(node2, node1, inn_num, weight)
        else:
            inn_num = self.get_inn_num(innovationhistory, node1, node2)
            new_conn = TFConnection(node1, node2, inn_num, weight)

        self.connections.append(new_conn)
        self.connect_nodes()

    def sort_nodes_by_layer(self) -> None:
        """
        Sorts the nodes by layer
        """
        self.nodes.sort(key=lambda x: x.layer)

    def clone(self) -> "TFGenome":
        """
        Returns a copy of the Genome
        """
        clone = TFGenome(self.num_in, self.num_out, True)
        for node in self.nodes:
            clone.nodes.append(node.clone())
        for connection in self.connections:
            clone.connections.append(connection.clone(clone.get_node(
                connection.node_in.num), clone.get_node(connection.node_out.num)))
        clone.fitness = self.fitness
        clone.unadjusted_fitness = self.unadjusted_fitness
        clone.best_score = self.best_score
        clone.layers = self.layers
        clone.connect_nodes()
        return clone

    def is_fully_connected(self) -> bool:
        """
        Checks if the Network is fully connected
        """
        num_nodes = {}
        for i in range(self.layers):
            num_nodes[i] = 0
        # Count the number of nodes per layers
        for node in self.nodes:
            num_nodes[node.layer] += 1

        max_connections = 0
        # For every pair of layers
        for key_i in range(self.layers-1):
            for key_j in range(key_i+1, self.layers):
                # Increase the max number of connections by the number of possible connections
                # Between the pair of layers
                max_connections += num_nodes[key_i] * num_nodes[key_j]

        return max_connections <= len(self.connections)

    def is_connected(self, node1: TFNode, node2: TFNode) -> bool:
        """
        Checks if there is a connection between the Nodes in the Genome.
        (Node) node1: The first node
        (Node) node2: The second node
        """
        for conn in self.connections:
            if conn.node_in == node1 and conn.node_out == node2:
                return True

            if conn.node_in == node2 and conn.node_out == node1:
                return True
        return False

    def get_node(self, num: int) -> TFNode:
        """
        Finds and retrieves a Node in the Genome. Else returns -1
        (int) num: The innovation number of the Node that is searched for
        """
        for node in self.nodes:
            if node.num == num:
                return node

        else:
            return -1

    def find_conn_idx(self, c: TFConnection) -> int:
        """
        Finds and retrieves the Index of a connection in the Genomes Connecton List. 
        Else returns -1
        (int) c: the Connection to search for
        """
        for idx, conn in enumerate(self.connections):
            if conn.get_innovation_num() == c.get_innovation_num():
                return idx
        # No connection with same innovation number found
        return -1

    def calc_weight(self) -> int:
        """
        Returns the computational weight of the network
        """
        return len(self.connections) + len(self.nodes)

    def get_inn_num(self, innovationhistory: List[ConnectionHistory], node_in: TFNode, node_out: TFNode):
        """
        Returns the innovation number for the mutation
        """
        is_new = True
        inn_num = innovationcounter.conn_id_count

        for hist in innovationhistory:
            if hist.matches(self, node_in, node_out):
                is_new = False
                inn_num = hist.inn_num
                break

        if is_new:
            numbers = []
            for conn in self.connections:
                numbers.append(conn.inn_num)

            innovationhistory.append(ConnectionHistory(
                node_in.num, node_out.num, inn_num, numbers))
            innovationcounter.conn_id_count += 1

        return inn_num
