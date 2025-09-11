"""
Этот модуль содержит классы-"сэмплеры", которые отвечают за генерацию
конфигураций атак на основе заданного пространства поиска.
"""
import random
from typing import Dict, Any, Union, List

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
        with open(search_space_path, 'r') as f:
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
            
            raise ValueError(f"Неизвестный тип сэмплирования: {sample_type}")

        sampled_node = {}
        for key, value in space_definition.items():
            sampled_node[key] = self._sample_from_space(value)
        return sampled_node