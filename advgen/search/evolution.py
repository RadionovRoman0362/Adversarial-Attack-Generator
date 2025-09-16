import random
import copy
from typing import List, Dict, Any, Tuple, Union

from .samplers import RandomSampler


class EvolutionarySampler:
    def __init__(self, search_space_path: str, population_size: int, mutation_rate: float, tournament_size: int,
                 norm: str, objectives: List = None, constraints: List = None, mutation_strategy: List[Dict] = None):
        self.random_sampler = RandomSampler(search_space_path)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.norm = norm
        self.population: List[Dict[str, Any]] = []
        self.objectives = objectives if objectives else []
        self.constraints = constraints if constraints else []
        self.mutation_strategy = mutation_strategy if mutation_strategy else []

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

    def _crossover_params(self, p1_params: Dict, p2_params: Dict, component_type: str, component_name: str) -> Dict:
        """Скрещивает параметры двух компонентов одного типа."""
        child_params = {}
        all_param_keys = set(p1_params.keys()) | set(p2_params.keys())

        for key in all_param_keys:
            p1_val = p1_params.get(key)
            p2_val = p2_params.get(key)

            if p1_val is None or p2_val is None:
                # Если у одного из родителей параметра нет, берем от того, у кого есть
                child_params[key] = p1_val if p1_val is not None else p2_val
                continue

            # Получаем спецификацию параметра, чтобы понять, как его скрещивать
            param_spec = self.random_sampler.get_param_spec(component_type, component_name, key, self.norm)
            param_type = param_spec.get('type')

            if param_type in ['range_float', 'range_int']:
                # Арифметический кроссовер (BLX-alpha)
                alpha = random.uniform(-0.5, 1.5)
                val = p1_val + alpha * (p2_val - p1_val)

                # Ограничиваем значение границами из спецификации
                val = max(param_spec['min'], min(param_spec['max'], val))

                if param_type == 'range_int':
                    child_params[key] = int(round(val))
                else:
                    child_params[key] = val

            elif param_type == 'choice':
                # Однородный кроссовер для категориальных
                child_params[key] = random.choice([p1_val, p2_val])

            elif param_type == 'fixed':
                # Для фиксированных значений просто берем любое
                child_params[key] = p1_val
            else:
                # Если тип неизвестен, просто берем от одного из родителей
                child_params[key] = random.choice([p1_val, p2_val])

        return child_params

    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Иерархический кроссовер. Скрещивает компоненты, и если компоненты
        одного типа - скрещивает их параметры.
        """
        child = {}
        all_component_keys = set(parent1.keys()) | set(parent2.keys())

        for key in all_component_keys:
            p1_comp = parent1.get(key)
            p2_comp = parent2.get(key)

            if p1_comp is None or p2_comp is None:
                child[key] = p1_comp if p1_comp is not None else p2_comp
                continue

            can_crossover_params = (
                isinstance(p1_comp, dict) and isinstance(p2_comp, dict) and
                'name' in p1_comp and 'name' in p2_comp and
                p1_comp['name'] == p2_comp['name'] and
                'params' in p1_comp and 'params' in p2_comp
            )

            if can_crossover_params and random.random() < 0.5:
                child_comp = copy.deepcopy(p1_comp)

                component_name = p1_comp['name']
                p1_params = p1_comp.get('params', {})
                p2_params = p2_comp.get('params', {})

                child_comp['params'] = self._crossover_params(p1_params, p2_params, key, component_name)
                child[key] = child_comp
            else:
                child[key] = copy.deepcopy(random.choice([p1_comp, p2_comp]))

        return child

    def _mutate_parameter(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """
        Выполняет "тонкую настройку": выбирает один параметр и немного его изменяет.
        """
        mutated_individual = copy.deepcopy(individual)

        # Выбираем компонент, у которого есть настраиваемые параметры
        components_with_params = [
            k for k, v in mutated_individual.items()
            if isinstance(v, dict) and 'params' in v and v['params']
        ]
        if not components_with_params:
            return mutated_individual  # Нечего мутировать

        component_type = random.choice(components_with_params)
        component_name = mutated_individual[component_type]['name']
        params = mutated_individual[component_type]['params']

        param_to_mutate = random.choice(list(params.keys()))

        # Получаем спецификацию параметра
        param_spec = self.random_sampler.get_param_spec(
            component_type, component_name, param_to_mutate, self.norm
        )
        param_type = param_spec.get('type')
        current_val = params[param_to_mutate]

        if param_type == 'range_float':
            # Гауссова мутация
            sigma = (param_spec['max'] - param_spec['min']) * 0.1  # 10% от диапазона
            new_val = current_val + random.normalvariate(0, sigma)
            # Ограничиваем значение границами
            new_val = max(param_spec['min'], min(param_spec['max'], new_val))
            params[param_to_mutate] = new_val

        elif param_type == 'range_int':
            # Сдвиг на +/- 1
            new_val = current_val + random.choice([-1, 1])
            new_val = max(param_spec['min'], min(param_spec['max'], new_val))
            params[param_to_mutate] = new_val

        elif param_type == 'choice':
            # Выбор другого значения из списка
            possible_values = param_spec['values']
            # Исключаем текущее значение, если есть другие варианты
            other_values = [v for v in possible_values if v != current_val]
            if other_values:
                params[param_to_mutate] = random.choice(other_values)

        return mutated_individual

    def _mutate_structural(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """
        Выполняет структурную мутацию: полностью заменяет один из компонентов.
        """
        mutated_individual = copy.deepcopy(individual)
        component_to_mutate = random.choice(list(mutated_individual.keys()))

        # Генерируем полный новый геном и берем оттуда только нужный компонент
        temp_sample = self.random_sampler.sample(self.norm)
        if component_to_mutate in temp_sample:
            mutated_individual[component_to_mutate] = temp_sample[component_to_mutate]

        return mutated_individual

    def _mutate_inter_norm(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """
        Агрессивная мутация: заменяет один нормо-зависимый компонент
        (кроме проектора) на компонент от другой нормы.
        """
        mutated_individual = copy.deepcopy(individual)

        # Определяем список доступных норм и нормо-зависимых компонентов
        all_norms = [b['name'] for b in self.random_sampler.space['norm_specific_components']['values']]
        other_norms = [n for n in all_norms if n != self.norm]

        if not other_norms:
            return self._mutate_structural(individual)  # Фоллбэк, если других норм нет

        # Компоненты, которые можно "одолжить" у другой нормы
        # Исключаем 'projector', чтобы не нарушить валидность атаки
        mutable_norm_components = [
            key for key in individual.keys()
            if key in ['initializer', 'updater', 'scheduler']
        ]
        if not mutable_norm_components:
            return self._mutate_structural(individual)  # Фоллбэк

        component_to_mutate = random.choice(mutable_norm_components)
        target_norm = random.choice(other_norms)

        # Генерируем геном для "чужой" нормы и берем оттуда компонент
        try:
            temp_sample = self.random_sampler.sample(target_norm)
            if component_to_mutate in temp_sample:
                print(f"Агрессивная мутация: {component_to_mutate} из нормы {self.norm} -> {target_norm}")
                mutated_individual[component_to_mutate] = temp_sample[component_to_mutate]
        except ValueError as e:
            # Если у другой нормы нет такого компонента, делаем фоллбэк
            print(f"Не удалось выполнить агрессивную мутацию: {e}. Выполняется структурная.")
            return self._mutate_structural(individual)

        return mutated_individual

    def _mutation(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """
        Основной оператор мутации, который вызывает один из подтипов
        в соответствии с вероятностями, заданными в конфигурации.
        """
        if random.random() >= self.mutation_rate:
            return individual

        if not self.mutation_strategy:
            return self._mutate_structural(individual)

        mutation_types = [s['type'] for s in self.mutation_strategy]
        probabilities = [s['probability'] for s in self.mutation_strategy]

        chosen_type = random.choices(mutation_types, weights=probabilities, k=1)[0]

        if chosen_type == "parameter":
            return self._mutate_parameter(individual)
        elif chosen_type == "structural":
            return self._mutate_structural(individual)
        elif chosen_type == "inter_norm":
            return self._mutate_inter_norm(individual)
        else:
            return self._mutate_structural(individual)

    @staticmethod
    def _dominates(ind1_fitness: tuple, ind2_fitness: tuple) -> bool:
        """Проверяет, доминирует ли индивид 1 над индивидом 2 (цели минимизируются)."""
        not_worse = all(x <= y for x, y in zip(ind1_fitness, ind2_fitness))
        strictly_better = any(x < y for x, y in zip(ind1_fitness, ind2_fitness))
        return not_worse and strictly_better

    def _non_dominated_sort(self, population_fitnesses: List[tuple]) -> List[List[int]]:
        """
        Выполняет быструю недоминируемую сортировку.
        Возвращает список фронтов, где каждый фронт - это список индексов индивидов.
        """
        n = len(population_fitnesses)
        dominating_counts = [0] * n
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]

        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(population_fitnesses[i], population_fitnesses[j]):
                    dominated_solutions[i].append(j)
                    dominating_counts[j] += 1
                elif self._dominates(population_fitnesses[j], population_fitnesses[i]):
                    dominated_solutions[j].append(i)
                    dominating_counts[i] += 1

            if dominating_counts[i] == 0:
                fronts[0].append(i)

        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p_idx in fronts[i]:
                for q_idx in dominated_solutions[p_idx]:
                    dominating_counts[q_idx] -= 1
                    if dominating_counts[q_idx] == 0:
                        next_front.append(q_idx)
            i += 1
            if next_front:
                fronts.append(next_front)
            else:
                break

        return fronts

    @staticmethod
    def _calculate_crowding_distance(front_indices: List[int], population_fitnesses: List[tuple]) -> Dict[
        int, float]:
        """
        Вычисляет расстояние скученности для одного фронта.
        Возвращает словарь {индекс_индивида: расстояние}.
        """
        if not front_indices:
            return {}

        num_individuals = len(front_indices)
        num_objectives = len(population_fitnesses[0])

        # Инициализируем расстояния для всех индивидов на фронте
        distances = {idx: 0.0 for idx in front_indices}

        # Создаем временную структуру для удобства
        front_fitnesses = {idx: population_fitnesses[idx] for idx in front_indices}

        for m in range(num_objectives):
            # Сортируем индивидов по m-й цели
            sorted_indices = sorted(front_indices, key=lambda idx: front_fitnesses[idx][m])

            # Крайним точкам присваиваем бесконечное расстояние
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')

            # Если на фронте меньше 3 точек, расстояние для внутренних не считаем
            if num_individuals <= 2:
                continue

            # Находим минимальное и максимальное значение цели для нормализации
            min_obj_val = front_fitnesses[sorted_indices[0]][m]
            max_obj_val = front_fitnesses[sorted_indices[-1]][m]

            # Если все значения по цели одинаковы, пропускаем, чтобы избежать деления на ноль
            if max_obj_val == min_obj_val:
                continue

            scale = max_obj_val - min_obj_val

            # Проходим по внутренним точкам
            for i in range(1, num_individuals - 1):
                prev_idx = sorted_indices[i - 1]
                next_idx = sorted_indices[i + 1]
                current_idx = sorted_indices[i]

                # Прибавляем нормализованное расстояние
                distances[current_idx] += (front_fitnesses[next_idx][m] - front_fitnesses[prev_idx][m]) / scale

        return distances

    def _get_constraint_violation(self, individual_results: Dict[str, Any]) -> float:
        """Вычисляет степень нарушения ограничений. 0, если все ОК."""
        violation = 0.0
        if not self.constraints:
            return violation

        for const in self.constraints:
            metric = const['metric']
            const_type = const['type']
            threshold = const['threshold']
            value = individual_results.get(metric)

            if value is None:
                # Назначаем большой штраф, если метрика отсутствует в результатах
                violation += 1e6
                continue

            if const_type == 'greater_than':
                if value < threshold:
                    # Нарушение пропорционально "недобору" до порога
                    violation += (threshold - value)
            elif const_type == 'less_than':
                if value > threshold:
                    violation += (value - threshold)

        return violation

    def _crowded_tournament_selection(
            self,
            population_info: List[dict],
            population_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Отбор родителей с учетом принципа Constraint-Domination.
        population_info содержит {'index', 'rank', 'distance'}.
        population_results содержит полные словари с метриками.
        """
        parents = []
        for _ in range(self.population_size):
            # Выбираем двух случайных конкурентов из популяции
            try:
                p1_idx, p2_idx = random.sample(range(len(population_info)), 2)
            except ValueError:
                # Если в популяции меньше 2 особей, просто дублируем первую
                p1_idx, p2_idx = 0, 0
                if not population_info: return []

            # Получаем всю информацию о конкурентах
            p1_info = population_info[p1_idx]
            p2_info = population_info[p2_idx]
            p1_results = population_results[p1_idx]
            p2_results = population_results[p2_idx]

            # Вычисляем нарушения для каждого
            p1_violation = self._get_constraint_violation(p1_results)
            p2_violation = self._get_constraint_violation(p2_results)

            winner_info = None
            # --- Логика Constraint-Domination ---
            if p1_violation == 0 and p2_violation > 0:
                # p1 валиден, p2 - нет. p1 побеждает.
                winner_info = p1_info
            elif p1_violation > 0 and p2_violation == 0:
                # p2 валиден, p1 - нет. p2 побеждает.
                winner_info = p2_info
            elif p1_violation > 0 and p2_violation > 0:
                # Оба невалидны. Побеждает тот, кто меньше нарушает.
                winner_info = p1_info if p1_violation < p2_violation else p2_info
            else:
                # Оба валидны. Используем стандартное сравнение NSGA-II (crowded-comparison operator).
                if p1_info['rank'] < p2_info['rank']:
                    winner_info = p1_info
                elif p1_info['rank'] > p2_info['rank']:
                    winner_info = p2_info
                elif p1_info['distance'] > p2_info['distance']: # Ранги равны, выбираем по расстоянию скученности
                    winner_info = p1_info
                else: # Ранги и расстояния равны, выбираем случайно
                    winner_info = random.choice([p1_info, p2_info])

            parents.append(self.population[winner_info['index']])
        return parents

    def evolve(self, fitnesses: Union[List[float], List[tuple]], population_results: List[Dict[str, Any]]):
        """Выполняет один цикл эволюции: отбор -> скрещивание -> мутация."""
        print("Эволюция нового поколения...")

        is_multiobjective = isinstance(fitnesses[0], tuple)
        if not is_multiobjective:
            parents = self._selection(fitnesses)
        else:
            fronts = self._non_dominated_sort(fitnesses)

            population_info = []
            for rank, front in enumerate(fronts):
                distances = self._calculate_crowding_distance(front, fitnesses)
                for idx in front:
                    population_info.append({
                        "index": idx,
                        "rank": rank,
                        "distance": distances[idx]
                    })

            parents = self._crowded_tournament_selection(population_info, population_results)

        new_population = []
        if not parents:
            print("Предупреждение: пул родителей пуст. Генерация случайных особей.")
            self.initialize_population()
            return

        for i in range(0, self.population_size, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1] if i + 1 < len(parents) else parents[0]

            child1 = self._crossover(parent1, parent2)
            child2 = self._crossover(parent2, parent1)

            new_population.append(self._mutation(child1))
            new_population.append(self._mutation(child2))

        self.population = new_population[:self.population_size]