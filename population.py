from __future__ import annotations
from typing import Any, Callable, List, Tuple
from node import Node
from connection import Connection
import random
import math
from genome import Genome
from species import Species


class Population:
    """
    Represents a population of genomes.
    """

    def __init__(self, size_in: int, size_out: int, fitness_criterion, size_pop: int = 50):
        self.fitness_criterion = fitness_criterion
        self.genomes: List[Genome] = []
        self.species: List[List[Genome]] = []
        self.innovationhistory = []

        self.best_genome = None
        self.best_score = 0
        self.global_best_score = 0
        self.gen = 1
        self.best_per_gen: List[Genome] = []
        self.species: List[Species] = []

        self.mass_extinct = False
        self.new_stage = False

        # Create the initial population of genomes
        for i in range(size_pop):
            genome = Genome(size_in, size_out, False)
            genome.connect_nodes()
            genome.mutate_genome(self.innovationhistory)
            self.genomes.append(genome)

    def set_best(self):
        """
        Handles the update of the best player
        """
        # The best species is the last one in the list, sorted in ascending fitness order
        best_species = self.species[-1]
        # The best player is last in the species list, also sorted in ascending order
        best_player: Genome = best_species.genomes[-1]

        if best_player.unadjusted_fitness >= self.best_score:
            # produce a clone to store for later evaluation
            clone = best_player.clone()
            self.best_per_gen.append(clone)
            self.best_score = best_player.unadjusted_fitness
            self.best_genome = clone

    def natural_selection(self):
        """
        Called at the end of a generation: Prepares the next generation
        """
        prev_best = self.genomes[-1]
        self.speciate()
        self.calc_fitness()
        self.sort_species()

        if self.mass_extinct:
            self.mass_extinction()
            self.mass_extinct = False

        self.cull_species()
        self.set_best()
        self.kill_stale_species()
        self.kill_bad_species()

        avg_sum = self.get_avg_fitness_sum()
        children = []
        for spec in self.species:
            # Add a clone of the best performer of the species
            children.append(spec.winner.clone())
            # Define the number of children a species is able to produce
            # NOTE: -1 because the winner of this generation is added
            num_children = math.floor(
                spec.avg_fitness / avg_sum * len(self.genomes))
            num_children -= 1
            # Create Offsprings to fill the number of children
            for i in range(num_children):
                children.append(spec.create_offspring(self.innovationhistory))
        # If not enough children, clone the previous best genome
        if len(children) < len(self.genomes):
            children.append(prev_best.clone())
        # If still not enough children due to flooring, add more from the best performing species
        while len(children) < len(self.genomes):
            children.append(
                self.species[-1].create_offspring(self.innovationhistory))

        # Set the new list of Genomes
        self.genomes.clear()
        for child in children:
            self.genomes.append(child.clone())

        self.gen += 1

    def speciate(self) -> None:
        """
        Categorizes the Population into species, based on their similarity to the representative of
        the species
        """
        # clean the list of Genomes for each species
        for spec in self.species:
            spec.genomes.clear()
        # Speciate each genome
        for genome in self.genomes:
            found_species = False
            for spec in self.species:
                if spec.same_species(genome):
                    spec.add_genome(genome)
                    found_species = True
                    break
            # If the Genome doesn't fit into any existing species, it gets its own
            if not found_species:
                new_species = Species(genome)
                self.species.append(new_species)

        # remove species without survivors
        self.species[:] = [
            spec for spec in self.species if len(spec.genomes) > 0]

    def calc_fitness(self) -> None:
        """
        Calculate the current fitness for every Genome
        """
        for genome in self.genomes:
            genome.fitness = self.fitness_criterion(genome)
            genome.unadjusted_fitness = genome.fitness

    def sort_species(self) -> None:
        """
        Sort the Species and the Genomes in each Species by fitness 
        """
        for spec in self.species:
            spec.sort_by_fitness()

        self.species.sort(key=lambda spec: spec.avg_fitness)

    def mass_extinction(self) -> None:
        """
        apart from the top 5 species, everyone dies
        """
        while len(self.species) > 5:
            self.species.remove(self.species[0])

    def cull_species(self):
        """
        Perform a culling for each species
        """
        for spec in self.species:
            spec.cull()
            spec.share_fitness()
            spec.set_average()

    def kill_stale_species(self):
        """
        Kills species that havent made improvements in 15 generations
        """
        # For every species apart from the top 2
        for spec in self.species[0:-2]:
            if spec.gens_wo_change > 15:
                self.species.remove(spec)

    def kill_bad_species(self):
        """
        Remove a species that performs too bad
        """
        avg_sum = self.get_avg_fitness_sum()

        for spec in self.species:
            # Check if the species would even receive an offspring in the next generation
            if spec.avg_fitness / avg_sum * len(self.genomes) < 1:
                self.species.remove(spec)

    def get_avg_fitness_sum(self) -> float:
        """
        Returns the sum of the average fitnesses of each species
        """
        avg_sum = 0
        for spec in self.species:
            avg_sum += spec.avg_fitness
        return avg_sum
