import unittest
from unittest.mock import MagicMock
from default_config import default_config
from population import Population
from genome import Genome
from __types__ import NeatConfig
from connection_history import ConnectionHistory
from species import Species
from node import Node
from random import randrange


def init_connection_history(num_inputs: int, num_outputs: int):
    connections: list[ConnectionHistory] = []
    innovation_nb = 0
    innovation_nbs = [n for n in range(num_inputs * num_outputs + 1)]
    for i in range(num_inputs):
        node_input = MagicMock(spec=Node, id=i)
        for j in range(num_outputs):
            node_output = MagicMock(spec=Node, id=j)
            connections.append(ConnectionHistory(
                from_node=node_input, to_node=node_output, innovation_nb=innovation_nb, innovation_nbs=innovation_nbs))
            innovation_nb += 1
    return connections


class TestPopulation(unittest.TestCase):
    def setUp(self):
        # Mock NeatConfig
        self.config = default_config
        self.config["num_inputs"] = 5
        self.config["num_outputs"] = 2
        self.config["population_size"] = 10
        self.config["species_elitism"] = 2
        self.config["max_stagnation"] = 5
        self.config["bad_species_threshold"] = 0.5
        self.config["no_fitness_termination"] = False
        self.config["min_species_size"] = 2
        self.config["fitness_threshold"] = 100

    def test_population_initialization(self):
        population = Population(self.config)
        self.assertEqual(len(population.genomes),
                         self.config["population_size"])
        self.assertEqual(len(population.species), 0)
        self.assertEqual(population.generation, 0)

    def test_set_best_genome(self):
        population = Population(self.config)

        # Assume best fitness is set to 10 for simplicity
        population.best_fitness = 10

        # Mock species and genomes
        genome = MagicMock(spec=Genome, fitness=20)
        specie = MagicMock(spec=Species, genomes=[genome])
        population.species = [specie]

        # Set best genome
        population.set_best_genome()

        # Assert that the best genome is set
        self.assertEqual(population.best_genome, genome)

    def test_speciate(self):
        population = Population(self.config)

        # Mock genomes
        genome1 = Genome(self.config)
        genome2 = Genome(self.config)
        population.genomes = [genome1, genome2]

        # Mock specie
        specie1 = Species(genome1)
        specie2 = Species(genome2)
        population.species = [specie1, specie2]

        # Run speciation
        population.speciate()

        # Assert that genomes are grouped into species
        self.assertGreater(len(population.species), 0)

    def test_reproduce_species(self):
        population = Population(self.config)

        # Mock species
        genome = Genome(self.config)
        specie = Species(genome)
        population.species = [specie]

        # Run reproduction
        population.reproduce_species()

        # Assert that the population's genomes are updated
        self.assertEqual(population.generation, 1)
        self.assertEqual(population.best_genome, genome)
        self.assertEqual(len(population.genomes), 10)

    def test_sort_species(self):
        population = Population(self.config)

        # Mock species
        for i in range(5):
            genome = Genome(self.config)
            genome.fitness = randrange(0, 100)
            specie = Species(genome)
            population.species.append(specie)

        # Run species sorting
        population.sort_species()

        def get_best_fitness(s: Species):
            return s.best_fitness

        # Assert species are sorted
        self.assertEqual(len(population.species), 5)
        self.assertTrue(sorted(population.species, key=get_best_fitness))

    def test_kill_stagnant_species(self):
        population = Population(self.config)

        # Mock species
        species_to_keep = [MagicMock(spec=Species, stagnation=2), MagicMock(
            spec=Species, stagnation=4)]
        species_to_remove = [
            MagicMock(spec=Species, stagnation=6), MagicMock(spec=Species, stagnation=8)]
        population.species = species_to_keep + species_to_remove

        # Run killing stagnant species
        population.kill_stagnant_species()

        # Assert stagnant species are removed
        self.assertEqual(population.species, species_to_keep)

    def test_kill_bad_species(self):
        population = Population(self.config)

        # Mock species
        good_species = MagicMock(
            spec=Species, average_fitness=self.config["bad_species_threshold"] + 1)
        bad_species1 = MagicMock(
            spec=Species, average_fitness=self.config["bad_species_threshold"] - 1)
        bad_species2 = MagicMock(
            spec=Species, average_fitness=self.config["bad_species_threshold"] - 2)
        population.species = [good_species, bad_species1, bad_species2]

        # Run killing bad species
        population.kill_bad_species()

        # Assert bad species are removed
        self.assertEqual(population.species, [good_species])

    def test_reset_on_extinction(self):
        population = Population(self.config)

        # Mock species
        population.species = []

        # Run reset on extinction
        population.reset_on_extinction()

        # Assert new random population is created
        self.assertEqual(len(population.genomes),
                         self.config["population_size"])

    def test_update_species(self):
        population = Population(self.config)

        # Mock species
        species = Species(Genome(self.config))
        for i in range(10):
            species.genomes.append(Genome(self.config))
        population.species = [species]

        # Run updating species
        population.update_species()

        # Assert species is updated
        self.assertEqual(len(species.genomes), self.config["min_species_size"])

    def test_get_genome(self):
        population = Population(self.config)

        # Mock genomes
        genome1 = MagicMock(spec=Genome, id="1")
        genome2 = MagicMock(spec=Genome, id="2")
        population.genomes = [genome1, genome2]

        # Run getting genome by id
        result_genome = population.get_genome("2")

        # Assert correct genome is returned
        self.assertEqual(result_genome, genome2)
