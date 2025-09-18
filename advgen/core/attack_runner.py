"""
Этот модуль содержит класс AttackRunner, который является ядром системы.

AttackRunner "собирает" состязательную атаку из различных компонентов
на основе предоставленной конфигурации и выполняет ее на батче данных.
"""

from typing import Dict, Any, List

import torch

from .model_wrapper import ModelWrapper
from ..components import create_component
from ..components.base import (
    Initializer, Loss, GradientComputer, Scheduler, Projector, UpdateRule
)


class AttackRunner:
    """
    Класс-оркестратор, который выполняет итеративную состязательную атаку.
    """

    def __init__(self, attack_config: Dict[str, Any], all_model_wrappers: List[ModelWrapper]):
        """
        Инициализирует AttackRunner, создавая все необходимые компоненты
        на основе конфигурационного словаря.

        :param attack_config: Словарь с полной конфигурацией атаки.
        """
        self.config = attack_config
        self.all_model_wrappers = all_model_wrappers if all_model_wrappers else []

        self.epsilon = attack_config['epsilon']
        self.steps = attack_config['steps']

        self.initializer: Initializer = self._create_comp('initializer')
        self.loss_fn: Loss = self._create_comp('loss')
        self.gradient_calc: GradientComputer = self._create_comp('gradient')
        self.scheduler: Scheduler = self._create_comp('scheduler')
        self.projector: Projector = self._create_comp('projector')
        self.updater: UpdateRule = self._create_comp('updater')

        self.components = [
            self.initializer, self.loss_fn, self.gradient_calc,
            self.scheduler, self.projector, self.updater
        ]

    def _create_comp(self, comp_type: str) -> Any:
        """Вспомогательная функция для создания компонента из конфига."""
        component_config = self.config[comp_type]
        return create_component(
            component_type=comp_type,
            config=component_config
        )

    def reset_components(self) -> None:
        """Сбрасывает внутреннее состояние всех компонентов."""
        for component in self.components:
            component.reset()

    def attack(
            self,
            surrogate_model_wrapper: ModelWrapper,
            images: torch.Tensor,
            labels: torch.Tensor,
            keep_graph: bool = False
    ) -> torch.Tensor:
        """
        Выполняет полный цикл состязательной атаки.

        :param surrogate_model_wrapper: Обертка над атакуемой моделью.
        :param images: Батч оригинальных изображений (тензор в [0, 1]).
        :param labels: Истинные метки для изображений.
        :param keep_graph: Сохранять граф для состязательного обучения.
        :return: Батч состязательных изображений.
        """
        original_images = images.clone().detach()

        self.reset_components()

        adv_images = self.initializer.initialize(original_images, self.epsilon)

        for i in range(self.steps):
            adv_images.requires_grad = True

            logits = surrogate_model_wrapper.get_logits(adv_images)
            loss = self.loss_fn(logits, labels)

            grad = self.gradient_calc.compute(
                surrogate_model=surrogate_model_wrapper,
                images=adv_images,
                labels=labels,
                loss_fn=self.loss_fn,
                all_models=self.all_model_wrappers
            )

            step_size = self.scheduler.get_step(
                current_step=i,
                total_steps=self.steps,
                loss=loss
            )

            adv_images = self.updater.update(adv_images, grad, step_size)

            adv_images = self.projector.project(
                adv_images=adv_images,
                original_images=original_images,
                epsilon=self.epsilon
            )

        return adv_images if keep_graph else adv_images.detach()

    @staticmethod
    def _format_config_recursively(config: Dict[str, Any]) -> str:
        """
        Рекурсивно форматирует конфигурацию компонента для красивого вывода.
        """
        if not isinstance(config, dict) or 'name' not in config:
            return str(config)

        name = config.get('name')
        params = config.get('params', {})

        args_list = []
        if params:
            params_str = ", ".join(f"{k}={v}" for k, v in params.items())
            args_list.append(params_str)

        if 'wrapped' in config:
            inner_str = AttackRunner._format_config_recursively(config['wrapped'])
            args_list.append(f"wrapped={inner_str}")

        args_str = ", ".join(args_list)
        return f"{name}({args_str})"

    def __repr__(self) -> str:
        """Представление объекта для вывода и отладки."""
        repr_str = "AttackRunner(\n"
        config_to_print = self.config.copy()

        attack_params = {
            'epsilon': config_to_print.pop('epsilon', 'N/A'),
            'steps': config_to_print.pop('steps', 'N/A'),
            'norm': config_to_print.pop('norm', 'N/A')
        }

        for comp_type, config in config_to_print.items():
            if isinstance(config, dict):
                formatted_str = self._format_config_recursively(config)
                repr_str += f"  {comp_type}: {formatted_str},\n"
            else:
                repr_str += f"  {comp_type}: {config},\n"
        for name, value in attack_params.items():
            repr_str += f"  {name}: {value},\n"

        repr_str += ")"
        return repr_str