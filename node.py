from __future__ import annotations
from random import random, choice
from numpy.random import uniform, normal
from __types__ import NeatConfig, ActivationFunctions
import activation_functions as activation_functions


class Node():
    """
    Node

    Represents a node in the NEAT algorithm, used within the context of a neural network.

    Attributes:
    - id (int): Unique identifier for the node.
    - input_sum (float): Current sum before activation.
    - output_value (float): After the activation function is applied.
    - output_connections (list[ConnectionGene]): List of connection genes representing connections from this node to others.
    - layer (int): Layer in the neural network to which the node belongs.
    - activation_function (ActivationFunctions): Activation function for the node.

    Methods:
    - get_activation_function(activation_function: ActivationFunctions) -> callable: Get the activation function based on the specified string.
    - engage() -> None: The node sends its output to the inputs of the nodes it's connected to.
    - mutate(config: NeatConfig, is_bias_node: bool = False) -> None: Mutate the node's properties based on the NEAT configuration.
    - is_connected_to(node: Node) -> bool: Check if this node is connected to the specified node.
    - clone() -> Node: Return a copy of this node.

    """

    def __init__(self, id: int, activation_function: ActivationFunctions, layer=0) -> None:
        """
        Initialize a Node instance.

        Args:
        - id (int): Unique identifier for the node.
        - activation_function (ActivationFunctions): Activation function for the node.
        - layer (int): Layer in the neural network to which the node belongs (default is 0).

        """
        from connection_gene import ConnectionGene
        self.id = id
        self.input_sum = 0  # current sum before activation
        self.output_value = 0  # after activation function is applied
        self.output_connections: list[ConnectionGene] = []
        self.layer = layer
        self.activation_function = activation_function

    def get_activation_function(self, activation_function: ActivationFunctions):
        """
        Get the activation function based on the specified string.

        Args:
        - activation_function (ActivationFunctions): String representing the activation function.

        Returns:
        - callable: The corresponding activation function.

        """
        if activation_function == "elu":
            return activation_functions.elu
        elif activation_function == "leaky_relu":
            return activation_functions.leaky_relu
        elif activation_function == "linear":
            return activation_functions.linear
        elif activation_function == "prelu":
            return activation_functions.prelu
        elif activation_function == "relu":
            return activation_functions.relu
        elif activation_function == "sigmoid":
            return activation_functions.sigmoid
        elif activation_function == "softmax":
            return activation_functions.softmax
        elif activation_function == "step":
            return activation_functions.step
        elif activation_function == "swish":
            return activation_functions.swish
        elif activation_function == "tanh":
            return activation_functions.tanh
        else:
            return activation_functions.sigmoid

    def engage(self) -> None:
        """
        The node sends its output to the inputs of the nodes it's connected to.

        """
        if self.layer != 0:
            activation = self.get_activation_function(self.activation_function)
            self.output_value = activation(self.input_sum)

        for c in self.output_connections:
            if c.enabled:
                c.to_node.input_sum += c.weight * self.output_value

    def mutate(self, config: NeatConfig, is_bias_node=False) -> None:
        """
        Mutate the node's properties based on the NEAT configuration.

        Args:
        - config (NeatConfig): NEAT configuration settings.
        - is_bias_node (bool): Flag indicating whether the node is a bias node (default is False).

        """
        if is_bias_node:
            if random() < config["bias_replace_rate"]:
                self.output_value = uniform(
                    config["bias_min_value"], config["bias_max_value"])

            elif random() < config["bias_mutate_rate"]:
                # otherwise slightly change it
                self.output_value += normal(config["bias_init_mean"],
                                            config["bias_init_stdev"]) / 50
                # keep weight between bounds
                if self.output_value > config["bias_max_value"]:
                    self.output_value = config["bias_max_value"]
                if self.output_value < config["bias_min_value"]:
                    self.output_value = config["bias_min_value"]

        if random() < config["activation_mutate_rate"]:
            activations_functions = ["step", "sigmoid", "tanh", "relu",
                                     "leaky_relu", "prelu", "elu", "softmax", "linear", "swish"]
            random_function = choice(activations_functions)
            self.activation_function = self.get_activation_function(
                random_function)

    def is_connected_to(self, node: Node) -> bool:
        """
        Check if this node is connected to the specified node.

        Args:
        - node (Node): The node to check for a connection.

        Returns:
        - bool: True if connected, False otherwise.

        """
        if node.layer == self.layer:
            return False

        if node.layer < self.layer:
            for c in node.output_connections:
                if c.to_node == self:
                    return True
        else:
            for c in self.output_connections:
                if c.to_node == node:
                    return True

        return False

    def clone(self) -> Node:
        """
        Return a copy of this node.

        Returns:
        - Node: A copy of this node.

        """
        return Node(self.id, self.activation_function, self.layer)
