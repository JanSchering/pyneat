from __future__ import annotations
from typing import Callable, List
from node import Node
from connection import Connection
import random
import math
from genome import Genome


class Species:
    """
    Represents a Species of Genomes.
    """

    def __init__(self, genome: Genome):
        self.genomes: List[Genome] = []
        self.best_fitness: float = 0
        self.winner: Genome = None
        self.avg_fitness = 0
        self.gens_wo_change = 0
        self.representative: Genome = None

        # Compatibility Distance factors
        self.excess_coeff = 1
        self.weight_diff_coeff = 0.5
        self.compat_thres = 3

        if genome:
            self.genomes.append(genome)
            # Its the only genome, so it is representative
            self.best_fitness = genome.fitness
            self.representative = genome
            self.winner = genome

    def same_species(self, genome: Genome) -> bool:
        count = self.count_excess_and_disjoint(genome)
        avg_weight_dif = self.calc_avg_weight_dif(genome)

        normalizer = genome.calc_weight() - 20
        normalizer = 1 if normalizer < 1 else normalizer

        compatibility = (self.excess_coeff*count)/normalizer + \
            self.weight_diff_coeff*avg_weight_dif
        return compatibility < self.compat_thres

    def add_genome(self, genome: Genome):
        self.genomes.append(genome)

    def count_excess_and_disjoint(self, genome: Genome) -> int:
        """"
        Counts the number of excess and disjoint Connection genes between it's representative
        and another Genome
        """
        matching_genes = 0
        # Compare the Connection Genes
        for conn1 in self.representative.connections:
            for conn2 in genome.connections:
                if conn1.get_innovation_num() == conn2.get_innovation_num():
                    matching_genes += 1
                    break

        # Number of excess and disjoint genes is the total number of genes - the matching genes
        # Note every match is a pair of genes, hence we double the count
        return len(self.representative.connections) + len(genome.connections) - 2*matching_genes

    def calc_avg_weight_dif(self, genome: Genome) -> float:
        """
        Calculates the average weight difference between the species' representative and another 
        Genome
        """
        matching_genes = 0
        total_diff = 0

        for conn1 in self.representative.connections:
            for conn2 in genome.connections:
                if conn1.get_innovation_num() == conn2.get_innovation_num():
                    matching_genes += 1
                    total_diff += abs(conn1.weight - conn2.weight)
                    break

        if matching_genes == 0:
            return 100

        return total_diff / matching_genes

    def sort_by_fitness(self) -> None:
        """
        Sorts the Genomes by their fitness
        """
        sorted_genomes = sorted(self.genomes, key=lambda x: x.fitness)

        # No changes, Set generations without change really high
        if self.genomes == sorted_genomes:
            self.gens_wo_change = 200
        # Check if there is a new pest-performing Genome
        elif sorted_genomes[-1].fitness > self.best_fitness:
            self.gens_wo_change = 0
            self.best_fitness = sorted_genomes[-1].fitness
            self.representative = sorted_genomes[-1]
            self.winner = sorted_genomes[-1]
        # some changes, but no new best performance
        else:
            self.gens_wo_change += 1

        self.genomes = sorted_genomes

    def set_average(self) -> None:
        """
        Calculates and updates the average fitness of the species
        """
        sum = 0
        for genome in self.genomes:
            sum += genome.fitness
        self.avg_fitness = sum / len(self.genomes)

    def create_offspring(self, innovationhistory) -> Genome:
        """
        Creates an offspring of the species by either asexual reproduction or crossover
        of two Genomes.
        """
        # 25% chance of asexual reproduction
        if random.uniform(0, 1) < 0.25:
            offspring = self.select_genome().clone()
        # 75% chance of crossover between two members of the species
        else:
            parent1 = self.select_genome()
            parent2 = self.select_genome()

            if parent1.fitness < parent2.fitness:
                offspring = parent2.produce_offspring(parent1)
            else:
                offspring = parent1.produce_offspring(parent2)

        offspring.mutate_genome(innovationhistory)
        return offspring

    def select_genome(self) -> Genome:
        """
        Select a Genome based on fitness
        """
        sum_fitness = 0
        for genome in self.genomes:
            sum_fitness += genome.fitness
        rand_thres = random.uniform(0, sum_fitness)

        running_sum = 0
        for idx in range(len(self.genomes)-1, -1, -1):
            genome = self.genomes[idx]
            running_sum += genome.fitness

            if running_sum > rand_thres:
                return genome

    def cull(self) -> None:
        """
        Performs a "culling" of the species: Kills the bottom half of the species
        """
        current_len = len(self.genomes)
        if current_len > 2:
            while len(self.genomes) > math.ceil(current_len/2):
                self.genomes.remove(self.genomes[0])

    def share_fitness(self) -> None:
        """
        Helps protecting new mutations by adjusting the fitness of each genome according to the size
        of the species
        """
        for genome in self.genomes:
            genome.fitness /= len(self.genomes)
