from __future__ import annotations
from typing import TYPE_CHECKING
from node import Node

if TYPE_CHECKING:  # pragma: no cover
    from genome import Genome

class ConnectionHistory():
    """
    Connection History

    Represents a connection history entry in the NEAT algorithm, tracking the innovation numbers
    and nodes involved in a connection.

    Attributes:
    - from_node (Node): The source node of the connection.
    - to_node (Node): The target node of the connection.
    - innovation_nb (int): The innovation number uniquely identifying this connection.
    - innovation_nbs (list[int]): List of innovation numbers from connections in the genome when this mutation first occurred.

    Methods:
    - matches(genome: Genome, from_node: Node, to_node: Node) -> bool: Checks if a given genome and connection nodes match the history entry.

    """

    def __init__(self, from_node: Node, to_node: Node, innovation_nb: int, innovation_nbs: "list[int]") -> None:
        """
        Initialize a ConnectionHistory instance.

        Args:
        - from_node (Node): The source node of the connection.
        - to_node (Node): The target node of the connection.
        - innovation_nb (int): The innovation number uniquely identifying this connection.
        - innovation_nbs (list[int]): List of innovation numbers from connections in the genome when this mutation first occurred.

        """
        self.from_node = from_node
        self.to_node = to_node
        self.innovation_nb = innovation_nb
        # the innovation numbers from the connections of the genome which first had this mutation
        self.innovation_nbs: list[int] = []
        # this represents the genome and allows us to test if another genomes is the same
        # this is before this connection was added
        self.innovation_nbs = innovation_nbs.copy()

    def matches(self, genome: Genome, from_node: Node, to_node: Node):
        """
        Check if a given genome and connection nodes match the history entry.

        Args:
        - genome (Genome): The genome to compare with the connection history.
        - from_node (Node): The source node of the connection to check.
        - to_node (Node): The target node of the connection to check.

        Returns:
        - bool: True if the genome and connection nodes match the history entry, False otherwise.

        """
        # if the number of connections are different then the genomes aren't the same
        if len(genome.genes) == len(self.innovation_nbs):
            if from_node.id == self.from_node.id and to_node.id == self.to_node.id:
                # next check if all the innovation numbers match from the genome
                for g in genome.genes:
                    if g.innovation_nb not in self.innovation_nbs:
                        return False

                # if reached this far then the innovationNumbers match the genes innovation numbers and
                # the connection is between the same nodes so it does match
                return True

        return False
