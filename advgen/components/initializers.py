"""
Этот модуль содержит конкретные реализации инициализаторов,
которые определяют начальную точку для состязательной атаки.

Все классы в этом файле наследуются от абстрактного класса `Initializer`
из модуля `base.py` и реализуют его метод `initialize`.
"""

import torch

from .base import Initializer


class ZeroInitializer(Initializer):
    """
    Нулевой инициализатор.
    Атака начинается непосредственно с оригинального изображения.
    Начальное возмущение равно нулю. Универсален для всех норм.
    """
    def initialize(self, images: torch.Tensor, epsilon: float) -> torch.Tensor:
        """
        Возвращает оригинальные изображения без изменений.

        :param images: Оригинальные изображения.
        :param epsilon: Параметр не используется, но присутствует для
                        совместимости с интерфейсом.
        :return: Тензор оригинальных изображений.
        """
        return images.clone().detach()


class RandomLinfInitializer(Initializer):
    """
    Случайный L-infinity инициализатор.
    Начинает атаку со случайной точки внутри L-infinity шара (куба)
    радиусом эпсилон от оригинального изображения. Использует равномерное распределение.

    Это стандартный подход для сильных L-inf атак (например, PGD), который помогает
    избежать локальных оптимумов и повышает вероятность успеха.
    """
    def initialize(self, images: torch.Tensor, epsilon: float) -> torch.Tensor:
        """
        Создает начальное возмущение в диапазоне [-epsilon, epsilon] и
        прибавляет его к оригинальным изображениям.

        :param images: Оригинальные изображения.
        :param epsilon: Максимальная величина возмущения для L-inf нормы.
        :return: Изображения с примененным случайным начальным возмущением,
                 ограниченные валидным диапазоном [0, 1].
        """
        adv_images = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
        adv_images = torch.clamp(adv_images, min=0, max=1)
        return adv_images.detach()


class RandomL2Initializer(Initializer):
    """
    Случайный L2 инициализатор.
    Начинает атаку со случайной точки, равномерно распределенной по объему
    L2-шара радиусом эпсилон от оригинального изображения.
    """

    def initialize(self, images: torch.Tensor, epsilon: float) -> torch.Tensor:
        """
        Создает случайное возмущение, нормализует его до единичной L2-нормы,
        масштабирует на математически корректный случайный радиус и
        прибавляет к изображениям.

        :param images: Оригинальные изображения.
        :param epsilon: Максимальная величина возмущения (радиус L2-шара).
        :return: Изображения с примененным случайным начальным возмущением,
                 ограниченные валидным диапазоном [0, 1].
        """
        batch_size = images.size(0)
        perturbation = torch.randn_like(images)

        flat_perturbation = perturbation.view(batch_size, -1)
        norms = torch.linalg.norm(flat_perturbation, ord=2, dim=1)
        flat_perturbation /= torch.clamp(norms.view(batch_size, -1), min=1e-12)

        # Математически корректное сэмплирование радиуса
        # для равномерного распределения по объему N-мерного шара.
        # U ~ Uniform(0,1) => R = epsilon * U^(1/N)
        dims = images.numel() / batch_size
        u = torch.rand(batch_size, device=images.device)
        random_radii = epsilon * torch.pow(u, 1.0 / dims)

        perturbation = flat_perturbation.view_as(images) * random_radii.view(batch_size, 1, 1, 1)

        adv_images = torch.clamp(images + perturbation, min=0, max=1)
        return adv_images.detach()