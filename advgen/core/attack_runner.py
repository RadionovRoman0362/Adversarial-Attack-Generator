"""
Этот модуль содержит класс AttackRunner, который является ядром системы.

AttackRunner "собирает" состязательную атаку из различных компонентов
на основе предоставленной конфигурации и выполняет ее на батче данных.
"""

from typing import Dict, Any

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

    def __init__(self, attack_config: Dict[str, Any]):
        """
        Инициализирует AttackRunner, создавая все необходимые компоненты
        на основе конфигурационного словаря.

        :param attack_config: Словарь с полной конфигурацией атаки.
                              Пример:
                              {
                                  "epsilon": 0.03,
                                  "steps": 40,
                                  "initializer": {"name": "random_linf"},
                                  "loss": {"name": "cross_entropy"},
                                  "gradient": {"name": "standard"},
                                  "scheduler": {"name": "fixed", "params": {"step_size": 0.01}},
                                  "projector": {"name": "linf"}
                              }
        """
        self.config = attack_config
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
        config = self.config[comp_type]
        return create_component(
            component_type=comp_type,
            name=config['name'],
            params=config.get('params', {})
        )

    def reset_components(self) -> None:
        """Сбрасывает внутреннее состояние всех компонентов."""
        for component in self.components:
            component.reset()

    def attack(
            self,
            model_wrapper: ModelWrapper,
            images: torch.Tensor,
            labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Выполняет полный цикл состязательной атаки.

        :param model_wrapper: Обертка над атакуемой моделью.
        :param images: Батч оригинальных изображений (тензор в [0, 1]).
        :param labels: Истинные метки для изображений.
        :return: Батч состязательных изображений.
        """
        original_images = images.clone().detach()

        self.reset_components()

        adv_images = self.initializer.initialize(original_images, self.epsilon)

        for i in range(self.steps):
            adv_images.requires_grad = True

            logits = model_wrapper.get_logits(adv_images)
            loss = self.loss_fn(logits, labels)

            grad = self.gradient_calc.compute(
                model=model_wrapper,
                images=adv_images,
                labels=labels,
                loss_fn=self.loss_fn
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

        return adv_images.detach()

    def __repr__(self) -> str:
        """Представление объекта для вывода и отладки."""
        repr_str = "AttackRunner(\n"
        for comp_type, config in self.config.items():
            if isinstance(config, dict):
                repr_str += f"  {comp_type}: {config.get('name')}({config.get('params', {})}),\n"
            else:
                repr_str += f"  {comp_type}: {config},\n"
        repr_str += ")"
        return repr_str