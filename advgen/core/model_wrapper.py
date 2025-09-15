"""
Этот модуль предоставляет класс-обертку ModelWrapper для моделей PyTorch.

ModelWrapper стандартизирует интерфейс взаимодействия с любой моделью,
автоматически выполняя две критически важные операции:
1. Перевод модели в режим инференса (`.eval()`).
2. Нормализацию входных изображений перед подачей их в модель.

Это позволяет основному коду атаки (AttackRunner) работать с "чистыми"
изображениями в диапазоне [0, 1], не задумываясь о специфических
требованиях предобработки конкретной модели.
"""

import logging
from typing import List, Tuple, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _validate_and_get_device(
        model: nn.Module,
        mean: Optional[List[float]],
        std: Optional[List[float]]
) -> torch.device:
    """Вспомогательная функция для валидации и определения устройства."""
    if not isinstance(model, nn.Module):
        raise TypeError(f"model должен быть экземпляром nn.Module, получено {type(model)}")
    if (mean is None) != (std is None):
        raise ValueError("mean и std должны быть либо оба предоставлены, либо оба None.")
    if mean is not None and len(mean) != len(std):
        raise ValueError("Длины mean и std должны совпадать.")

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
        logger.warning("Модель не имеет параметров. Устройство по умолчанию - CPU.")
    return device


class ModelWrapper(nn.Module):
    """
    Обертка для nn.Module, которая добавляет автоматическую нормализацию входа.
    """

    def __init__(
            self,
            model: nn.Module,
            mean: Optional[Tuple[float, float, float] or List[float]] = None,
            std: Optional[Tuple[float, float, float] or List[float]] = None
    ):
        """
        :param model: Модель PyTorch для атаки.
        :param mean: (Опционально) Средние значения для нормализации. Если None,
                     нормализация не применяется.
        :param std: (Опционально) Стандартные отклонения для нормализации. Если None,
                    нормализация не применяется.
        """
        super().__init__()

        self.device = _validate_and_get_device(model, mean, std)

        self.model = model.to(self.device)

        self.use_normalization = mean is not None
        if self.use_normalization:
            self.register_buffer(
                'norm_mean',
                torch.tensor(mean, device=self.device).view(1, -1, 1, 1)
            )
            self.register_buffer(
                'norm_std',
                torch.tensor(std, device=self.device).view(1, -1, 1, 1)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Выполняет прямой проход: сначала нормализует вход, затем пропускает через модель.

        :param x: Входной тензор изображений. Ожидается, что значения
                  находятся в диапазоне [0, 1].
        :return: Выходные логиты модели.
        """
        if x.device != self.device:
            x = x.to(self.device)

        if self.use_normalization:
            x = (x - self.norm_mean) / self.norm_std

        return self.model(x)

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Основной метод для вызова из AttackRunner. Является псевдонимом для forward().

        :param x: Входной тензор изображений в диапазоне [0, 1].
        :return: Выходные логиты модели.
        """
        return self.forward(x)

    def train(self, mode: bool = True):
        """Переключает обертку и внутреннюю модель в режим обучения."""
        super().train(mode)
        self.model.train(mode)
        return self

    def eval(self):
        """Переключает обертку и внутреннюю модель в режим оценки."""
        return self.train(False)

    def __repr__(self) -> str:
        """Представление объекта для вывода и отладки."""
        original_model_name = self.model.__class__.__name__
        if self.use_normalization:
            mean_list = self.norm_mean.flatten().tolist()
            std_list = self.norm_std.flatten().tolist()
            return (f"ModelWrapper(model={original_model_name}, device={self.device}, "
                    f"mean={[f'{m:.3f}' for m in mean_list]}, "
                    f"std={[f'{s:.3f}' for s in std_list]})")
        else:
            return f"ModelWrapper(model={original_model_name}, device={self.device}, normalization=False)"