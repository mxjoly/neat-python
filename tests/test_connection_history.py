import unittest
from default_config import default_config
from node import Node
from genome import Genome
from connection_history import ConnectionHistory
from connection_gene import ConnectionGene
import activation_functions


class TestConnectionHistory(unittest.TestCase):
    def setUp(self):
        self.config = default_config
        self.genome = Genome(self.config)
        self.from_node = Node(1, activation_functions.sigmoid, 1)
        self.to_node = Node(2, activation_functions.sigmoid, 2)
        self.innovation_nbs = [1]
        self.genome.genes = [
            ConnectionGene(self.from_node, self.to_node, 0.5, 1, True)
        ]
        self.connection_history = ConnectionHistory(
            self.from_node, self.to_node, 1, self.innovation_nbs)

    def test_init(self):
        self.assertEqual(self.connection_history.from_node, self.from_node)
        self.assertEqual(self.connection_history.to_node, self.to_node)
        self.assertEqual(self.connection_history.innovation_nbs,
                         self.innovation_nbs)

    def test_matches_with_existing_connection(self):
        result = self.connection_history.matches(
            self.genome, self.from_node, self.to_node)
        self.assertTrue(result)

    def test_matches_with_non_existing_connection(self):
        node = Node(3, activation_functions.relu, 2)
        result = self.connection_history.matches(
            self.genome, self.from_node, node)
        self.assertFalse(result)

    def test_matches_with_not_same_genome(self):
        connection_history = ConnectionHistory(
            self.from_node, self.to_node, 1, [2])
        result = connection_history.matches(
            self.genome, self.from_node, self.to_node)
        self.assertFalse(result)
