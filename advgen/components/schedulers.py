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


class WarmupCosineDecayScheduler(Scheduler):
    """
    Планировщик, который сначала линейно увеличивает шаг ("прогрев"),
    а затем уменьшает его по косинусоиде.
    """
    def __init__(self, max_step_size: float, warmup_steps: int):
        super().__init__()
        if max_step_size <= 0:
            raise ValueError("Максимальный размер шага должен быть положительным.")
        if warmup_steps < 0:
            raise ValueError("Количество шагов для прогрева не может быть отрицательным.")
        self.max_step_size = max_step_size
        self.warmup_steps = warmup_steps

    def get_step(self, current_step: int, total_steps: int, **kwargs) -> float:
        """
        Возвращает размер шага в зависимости от фазы: прогрев или затухание.
        """
        if self.warmup_steps > 0 and current_step < self.warmup_steps:
            # Фаза прогрева: линейный рост от 0 до max_step_size
            warmup_factor = (current_step + 1) / self.warmup_steps
            return self.max_step_size * warmup_factor
        else:
            # Фаза затухания по косинусоиде
            # Корректируем шаги, чтобы косинус начинался после прогрева
            effective_total_steps = total_steps - self.warmup_steps
            effective_current_step = current_step - self.warmup_steps

            if effective_total_steps <= 1:
                return self.max_step_size

            cosine_decay = 0.5 * (1 + math.cos(math.pi * effective_current_step / (effective_total_steps - 1)))
            return self.max_step_size * cosine_decay


class CyclicLRStepScheduler(Scheduler):
    """
    Планировщик, который циклически изменяет размер шага между
    base_step_size и max_step_size.
    """
    def __init__(self, base_step_size: float, max_step_size: float, cycle_steps: int):
        super().__init__()
        if base_step_size < 0 or max_step_size <= base_step_size:
            raise ValueError("Параметры шага некорректны: 0 <= base_step_size < max_step_size.")
        if cycle_steps <= 0:
            raise ValueError("Длина цикла должна быть положительной.")

        self.base_step_size = base_step_size
        self.max_step_size = max_step_size
        self.step_range = max_step_size - base_step_size
        self.cycle_steps = cycle_steps
        self.half_cycle = cycle_steps / 2.0

    def get_step(self, current_step: int, total_steps: int, **kwargs) -> float:
        """
        Вычисляет размер шага на основе текущей позиции в цикле.
        """
        cycle_position = current_step % self.cycle_steps

        cycle_multiplier = 1 - abs(cycle_position / self.half_cycle - 1)

        return self.base_step_size + self.step_range * cycle_multiplier


class PlateauReduceStepScheduler(Scheduler):
    """
    Уменьшает размер шага, когда функция потерь выходит на плато.
    Атака максимизирует loss, поэтому "улучшение" - это его рост.
    """
    def __init__(self, initial_step_size: float, factor: float = 0.5, patience: int = 5, min_step_size: float = 1e-6):
        super().__init__()
        if not (0 < factor < 1):
            raise ValueError("Фактор уменьшения должен быть в (0, 1).")
        if patience < 0:
            raise ValueError("Терпение не может быть отрицательным.")

        self.step_size = initial_step_size
        self.initial_step_size = initial_step_size
        self.factor = factor
        self.patience = patience
        self.min_step_size = min_step_size

        # Внутреннее состояние
        self.patience_counter = 0
        self.best_loss = -float('inf')

    def get_step(self, current_step: int, total_steps: int, **kwargs) -> float:
        """
        Проверяет условие плато и при необходимости уменьшает размер шага.
        """
        loss_tensor: Optional[torch.Tensor] = kwargs.get("loss")
        if loss_tensor is None:
            # Если loss не предоставлен, работаем как FixedStepScheduler
            return self.step_size

        current_loss = loss_tensor.item()

        # Считаем, что на первом шаге всегда есть "улучшение"
        if current_step == 0:
            self.best_loss = current_loss
            return self.step_size

        # "Улучшение" - это рост loss. Добавляем небольшой допуск (1e-4).
        if current_loss > self.best_loss + 1e-4:
            self.best_loss = current_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        if self.patience_counter >= self.patience:
            # Плато достигнуто, уменьшаем шаг
            new_step_size = self.step_size * self.factor
            if new_step_size >= self.min_step_size:
                print(
                    f"Плато достигнуто на шаге {current_step}. Уменьшение шага: {self.step_size:.6f} -> {new_step_size:.6f}")
                self.step_size = new_step_size
            # Сбрасываем счетчик, чтобы дать новому шагу время поработать
            self.patience_counter = 0

        return self.step_size

    def reset(self):
        """Сбрасывает внутреннее состояние планировщика."""
        self.step_size = self.initial_step_size
        self.patience_counter = 0
        self.best_loss = -float('inf')