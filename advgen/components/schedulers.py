"""
Этот модуль содержит конкретные реализации планировщиков размера шага.
Планировщик определяет, как изменяется "скорость" атаки (размер шага)
на протяжении итераций.

Все классы наследуются от `Scheduler` из `base.py`.
"""
import math
import torch
from typing import List, Optional

from .base import Scheduler


class FixedStepScheduler(Scheduler):
    """
    Использует фиксированный, заранее заданный размер шага на всех итерациях.
    """
    def __init__(self, step_size: float):
        super().__init__()
        if step_size <= 0:
            raise ValueError("Размер шага должен быть положительным.")
        self.step_size = step_size

    def get_step(self, current_step: int, total_steps: int, **kwargs) -> float:
        """
        Возвращает фиксированный размер шага.

        :param current_step: Номер текущей итерации (не используется).
        :param total_steps: Общее количество итераций (не используется).
        :return: Размер шага.
        """
        return self.step_size


class LinearDecayStepScheduler(Scheduler):
    """
    Линейно уменьшает размер шага от начального до нуля.
    """
    def __init__(self, initial_step_size: float):
        super().__init__()
        if initial_step_size <= 0:
            raise ValueError("Начальный размер шага должен быть положительным.")
        self.initial_step_size = initial_step_size

    def get_step(self, current_step: int, total_steps: int, **kwargs) -> float:
        if total_steps <= 1:
            return self.initial_step_size
        decay = 1.0 - (current_step / (total_steps - 1))
        return self.initial_step_size * decay


class CosineAnnealingStepScheduler(Scheduler):
    """
    Уменьшает размер шага от начального до нуля по косинусоиде.
    """
    def __init__(self, initial_step_size: float):
        super().__init__()
        if initial_step_size <= 0:
            raise ValueError("Начальный размер шага должен быть положительным.")
        self.initial_step_size = initial_step_size

    def get_step(self, current_step: int, total_steps: int, **kwargs) -> float:
        """
        Вычисляет размер шага для текущей итерации по формуле косинусного затухания.
        """
        if total_steps <= 1:
            return self.initial_step_size
        cosine_decay = 0.5 * (1 + math.cos(math.pi * current_step / (total_steps - 1)))
        return self.initial_step_size * cosine_decay


class ExponentialDecayStepScheduler(Scheduler):
    """
    Уменьшает размер шага экспоненциально с каждой итерацией.
    """
    def __init__(self, initial_step_size: float, gamma: float):
        super().__init__()
        if not (0 < gamma <= 1):
            raise ValueError("Фактор затухания gamma должен быть в (0, 1].")
        self.initial_step_size = initial_step_size
        self.gamma = gamma

    def get_step(self, current_step: int, total_steps: int, **kwargs) -> float:
        """
        Вычисляет размер шага как initial_step_size * (gamma ** current_step).
        """
        return self.initial_step_size * (self.gamma ** current_step)


class MultiStepDecayStepScheduler(Scheduler):
    """
    Уменьшает размер шага на заданных "вехах" (milestones).
    """
    def __init__(self, initial_step_size: float, milestones: List[int], gamma: float):
        super().__init__()
        self.initial_step_size = initial_step_size
        self.milestones = sorted(milestones)
        self.gamma = gamma
        if not (0 < gamma < 1):
            raise ValueError("Фактор затухания gamma должен быть в (0, 1).")

    def get_step(self, current_step: int, total_steps: int, **kwargs) -> float:
        """
        Уменьшает шаг, если текущая итерация пересекла одну из "вех".
        """
        power = sum(1 for milestone in self.milestones if current_step >= milestone)
        return self.initial_step_size * (self.gamma ** power)


class AdaptiveStepScheduler(Scheduler):
    """
    Адаптивно изменяет размер шага на основе истории значений функции потерь.
    Реализует логику из Auto-PGD (APGD).
    """
    def __init__(self, initial_step_size: float, window_size: int = 20, decrease_factor: float = 0.5):
        super().__init__()
        self.step_size = initial_step_size
        self.initial_step_size = initial_step_size
        self.window_size = window_size
        self.decrease_factor = decrease_factor
        self.loss_history: List[float] = []

    def get_step(self, current_step: int, total_steps: int, **kwargs) -> float:
        """
        Проверяет условие уменьшения шага и возвращает текущий размер шага.
        """
        loss: Optional[torch.Tensor] = kwargs.get("loss")
        if loss is None:
            return self.step_size

        self.loss_history.append(loss.item())

        w = self.window_size
        if current_step > w:
            # Условие 1: нет значительного прогресса за последние w шагов
            cond1 = (self.loss_history[current_step - 1] > self.loss_history[current_step - w - 1] - 1e-6)
            # Условие 2: обнаружен "пик" в значении loss, что может указывать на
            # "перепрыгивание" через оптимум
            cond2 = ((self.loss_history[current_step - 1] - self.loss_history[current_step - 2]) < -1e-6) and \
                    ((self.loss_history[current_step - 2] - self.loss_history[current_step - 3]) > -1e-6)

            if cond1 or cond2:
                self.step_size *= self.decrease_factor
                self.loss_history = []

        return self.step_size

    def reset(self):
        """Сбрасывает историю и размер шага до начальных значений."""
        self.loss_history = []
        self.step_size = self.initial_step_size