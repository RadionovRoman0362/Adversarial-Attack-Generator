"""
Этот модуль содержит классы-"сэмплеры", которые отвечают за генерацию
конфигураций атак на основе заданного пространства поиска.
"""
import random
from typing import Dict, Any, Union, List

import optuna
from optuna.trial import Trial

import numpy as np
import yaml


class RandomSampler:
    """
    Генерирует случайные, но валидные конфигурации атак на основе
    сложного, иерархического пространства поиска, определенного в YAML-файле.
    """

    def __init__(self, search_space_path: str):
        """
        Инициализирует сэмплер, загружая и парся пространство поиска.

        :param search_space_path: Путь к YAML-файлу с пространством поиска.
        """
        with open(search_space_path, 'r', encoding='utf-8') as f:
            self.space = yaml.safe_load(f)

    def sample(self, norm: str) -> Dict[str, Any]:
        """
        Создает одну полную, случайную конфигурацию атаки для ЗАДАННОЙ НОРМЫ.

        :param norm: Строка, указывающая норму ('linf', 'l2' и т.д.),
                     для которой нужно сгенерировать компоненты.
        :return: Словарь с конфигурацией, готовый для передачи в AttackRunner.
        """
        final_config: Dict[str, Any] = {}

        general_components_space = self.space.get('components', {})
        final_config.update(self._sample_from_space(general_components_space))

        norm_specific_values = self.space['norm_specific_components']['values']

        target_norm_block = None
        for block in norm_specific_values:
            if block['name'] == norm:
                target_norm_block = block
                break

        if target_norm_block is None:
            raise ValueError(f"Блок для нормы '{norm}' не найден в search_space.yaml")

        norm_components_space = target_norm_block.get('components', {})
        final_config.update(self._sample_from_space(norm_components_space))

        return final_config

    def _sample_from_space(self, space_definition: Any) -> Any:
        """
        Рекурсивно обходит пространство поиска и сэмплирует значения.
        """
        if not isinstance(space_definition, dict):
            return space_definition

        if 'type' in space_definition:
            sample_type = space_definition['type']

            if sample_type == 'fixed':
                return self._sample_from_space(space_definition['value'])
            if sample_type == 'range_int':
                return random.randint(space_definition['min'], space_definition['max'])
            if sample_type == 'range_float':
                log_sample = space_definition.get('log', False)
                if log_sample:
                    log_min = np.log(space_definition['min'])
                    log_max = np.log(space_definition['max'])
                    return float(np.exp(random.uniform(log_min, log_max)))
                return random.uniform(space_definition['min'], space_definition['max'])
            if sample_type == 'choice':
                chosen_option = random.choice(space_definition['values'])
                return self._sample_from_space(chosen_option)
            if sample_type == 'composite':
                max_depth = space_definition.get('max_depth', 3)
                num_decorators = random.randint(0, max_depth)
                terminal_config = self._sample_from_space(space_definition['terminal_components'])

                if num_decorators == 0:
                    return terminal_config

                head_config = self._sample_from_space(space_definition['decorator_components'])
                current_node = head_config

                for _ in range(num_decorators - 1):
                    next_decorator_config = self._sample_from_space(space_definition['decorator_components'])
                    current_node['wrapped'] = next_decorator_config
                    current_node = next_decorator_config

                current_node['wrapped'] = terminal_config

                return head_config
            
            raise ValueError(f"Неизвестный тип сэмплирования: {sample_type}")

        sampled_node = {}
        for key, value in space_definition.items():
            sampled_node[key] = self._sample_from_space(value)
        return sampled_node

    def get_param_spec(self, component_type: str, component_name: str, param_name: str, norm: str) -> Dict[str, Any]:
        """
        Находит и возвращает спецификацию (описание) для конкретного параметра.
        """
        if component_type == 'gradient':
            grad_space = self.space['components']['gradient']

            for comp_spec in grad_space['terminal_components']['values']:
                if comp_spec['name'] == component_name:
                    if 'params' in comp_spec and param_name in comp_spec['params']:
                        return comp_spec['params'][param_name]

            for comp_spec in grad_space['decorator_components']['values']:
                if comp_spec['name'] == component_name:
                    if 'params' in comp_spec and param_name in comp_spec['params']:
                        return comp_spec['params'][param_name]

        try:
            all_component_options = self.space.get('components', {})
            if component_type in all_component_options:
                component_options = all_component_options[component_type]['values']
                component_spec = next(c for c in component_options if c['name'] == component_name)
                return component_spec['params'][param_name]
        except (KeyError, StopIteration):
            pass

        try:
            norm_blocks = self.space['norm_specific_components']['values']
            norm_block = next(b for b in norm_blocks if b['name'] == norm)

            component_options = norm_block['components'][component_type]['values']
            component_spec = next(c for c in component_options if c['name'] == component_name)
            return component_spec['params'][param_name]
        except (KeyError, StopIteration):
            raise ValueError(f"Спецификация для параметра {component_type}.{component_name}.{param_name} "
                             f"для нормы {norm} не найдена в search_space.yaml")


class OptunaSampler:
    """
    "Умный" сэмплер, который использует Optuna (и байесовскую оптимизацию)
    для выбора следующей конфигурации на основе предыдущих результатов.
    """

    def __init__(self, search_space_path: str):
        """
        Инициализирует сэмплер, загружая пространство поиска.
        :param search_space_path: Путь к YAML-файлу с пространством поиска.
        """
        with open(search_space_path, 'r') as f:
            self.space = yaml.safe_load(f)

    def sample(self, trial: Trial, norm: str) -> Dict[str, Any]:
        """
        Предлагает (suggest) одну конфигурацию атаки, используя объект `trial` из Optuna.
        :param trial: Объект optuna.Trial, который отслеживает текущий прогон.
        :param norm: Строка, указывающая норму ('linf', 'l2', и т.д.).
        :return: Словарь с конфигурацией, готовый для передачи в AttackRunner.
        """
        final_config: Dict[str, Any] = {}

        general_components_space = self.space.get('components', {})
        for key, value in general_components_space.items():
            if key == 'gradient':
                optuna_grad_space = self.space.get('gradient_optuna')
                if not optuna_grad_space:
                    raise ValueError(
                        "Для OptunaSampler в search_space.yaml должен быть определен блок 'gradient_optuna'")

                final_config[key] = self._sample_flattened_composite(trial, "gradient_", optuna_grad_space)
            else:
                final_config[key] = self._sample_from_space(trial, f"{key}_", value)

        norm_specific_values = self.space['norm_specific_components']['values']
        target_norm_block = next((b for b in norm_specific_values if b['name'] == norm), None)

        if target_norm_block is None:
            raise ValueError(f"Блок для нормы '{norm}' не найден в search_space.yaml")

        norm_components_space = target_norm_block.get('components', {})
        for key, value in norm_components_space.items():
            final_config[key] = self._sample_from_space(trial, f"{norm}_{key}_", value)

        return final_config

    def _sample_flattened_composite(self, trial: Trial, prefix: str, space_definition: Dict[str, Any]) -> Dict[
        str, Any]:
        """
        Сэмплирует композитный компонент по принципу "плоских слотов" для Optuna.
        """
        max_depth = space_definition.get('max_depth', 3)

        all_components = space_definition['decorator_options']['values'] + space_definition['terminal_options'][
            'values']
        all_params = {}

        for comp_spec in all_components:
            comp_name = comp_spec['name']
            if comp_name == 'none':
                continue
            if 'params' in comp_spec:
                all_params[comp_name] = self._sample_from_space(trial, f"{prefix}{comp_name}_",
                                                                {'params': comp_spec['params']})

        decorator_choices = [spec['name'] for spec in space_definition['decorator_options']['values']]
        selected_components = []
        for i in range(max_depth):
            slot_name = f"{prefix}slot_{i + 1}"
            chosen_decorator = trial.suggest_categorical(slot_name, decorator_choices)
            if chosen_decorator != 'none':
                selected_components.append(chosen_decorator)

        terminal_choices = [spec['name'] for spec in space_definition['terminal_options']['values']]
        chosen_terminal = trial.suggest_categorical(f"{prefix}terminal", terminal_choices)
        selected_components.append(chosen_terminal)

        chain = []
        for comp_name in selected_components:
            node = {'name': comp_name}
            if comp_name in all_params and 'params' in all_params[comp_name]:
                node['params'] = all_params[comp_name]['params']
            chain.append(node)

        if not chain:
            return {}

        head = chain[0]
        current_node = head
        for i in range(1, len(chain)):
            current_node['wrapped'] = chain[i]
            current_node = current_node['wrapped']

        return head

    def _sample_from_space(self, trial: Trial, prefix: str, space_definition: Any) -> Any:
        """
        Рекурсивно обходит пространство и предлагает параметры для Optuna.
        """
        if not isinstance(space_definition, dict):
            return space_definition

        if 'type' in space_definition:
            sample_type = space_definition['type']

            param_name = f"{prefix}{space_definition.get('param_name', '')}"

            if sample_type == 'fixed':
                return self._sample_from_space(trial, prefix, space_definition['value'])

            if sample_type == 'range_int':
                return trial.suggest_int(param_name, space_definition['min'], space_definition['max'])

            if sample_type == 'range_float':
                return trial.suggest_float(
                    param_name,
                    space_definition['min'],
                    space_definition['max'],
                    log=space_definition.get('log', False)
                )

            if sample_type == 'choice':
                options = [opt['name'] if isinstance(opt, dict) else opt for opt in space_definition['values']]
                chosen_name = trial.suggest_categorical(param_name, options)

                chosen_option = next((opt for opt in space_definition['values'] if
                                      (isinstance(opt, dict) and opt['name'] == chosen_name) or (opt == chosen_name)),
                                     None)
                return self._sample_from_space(trial, f"{prefix}{chosen_name}_", chosen_option)

            raise ValueError(f"Неизвестный тип сэмплирования: {sample_type}")

        sampled_node = {}
        for key, value in space_definition.items():
            new_prefix = f"{prefix}{key}_" if key != 'params' else prefix

            if key == 'params':
                sampled_params = {}
                for param_key, param_value in value.items():
                    param_value['param_name'] = param_key
                    sampled_params[param_key] = self._sample_from_space(trial, new_prefix, param_value)
                sampled_node[key] = sampled_params
            else:
                sampled_node[key] = self._sample_from_space(trial, new_prefix, value)
        return sampled_node