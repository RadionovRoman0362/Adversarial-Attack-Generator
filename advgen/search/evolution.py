import random
import copy
from typing import List, Dict, Any, Tuple

from .samplers import RandomSampler


class EvolutionarySampler:
    def __init__(self, search_space_path: str, population_size: int, mutation_rate: float, tournament_size: int,
                 norm: str):
        self.random_sampler = RandomSampler(search_space_path)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.norm = norm
        self.population: List[Dict[str, Any]] = []

    def initialize_population(self):
        """Создает начальную популяцию, переиспользуя RandomSampler."""
        print("Инициализация начальной популяции...")
        self.population = [self.random_sampler.sample(self.norm) for _ in range(self.population_size)]

    def _selection(self, fitness_scores: List[float]) -> List[Dict[str, Any]]:
        """Турнирный отбор."""
        parents = []
        for _ in range(self.population_size):
            tournament_indices = random.sample(range(self.population_size), self.tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_index_in_tournament = tournament_fitness.index(max(tournament_fitness))
            winner_index_in_population = tournament_indices[winner_index_in_tournament]
            parents.append(self.population[winner_index_in_population])
        return parents

    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Однородное скрещивание на уровне компонентов."""
        child = {}
        all_component_keys = parent1.keys()
        for key in all_component_keys:
            if random.random() < 0.5:
                child[key] = copy.deepcopy(parent1[key])
            else:
                child[key] = copy.deepcopy(parent2[key])
        return child

    def _mutation(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """Мутация через пересэмплирование одного из компонентов."""
        mutated_individual = copy.deepcopy(individual)
        if random.random() < self.mutation_rate:
            component_to_mutate = random.choice(list(mutated_individual.keys()))

            temp_sample = self.random_sampler.sample(self.norm)
            if component_to_mutate in temp_sample:
                mutated_individual[component_to_mutate] = temp_sample[component_to_mutate]
        return mutated_individual

    def evolve(self, fitness_scores: List[float]):
        """Выполняет один цикл эволюции: отбор -> скрещивание -> мутация."""
        print("Эволюция нового поколения...")
        parents = self._selection(fitness_scores)

        new_population = []
        for i in range(0, self.population_size, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1] if i + 1 < self.population_size else parents[i]

            child1 = self._crossover(parent1, parent2)
            child2 = self._crossover(parent2, parent1)

            new_population.append(self._mutation(child1))
            new_population.append(self._mutation(child2))

        self.population = new_population[:self.population_size]