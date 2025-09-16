"""
Этот модуль содержит конкретные реализации методов вычисления и/или
модификации градиента. Градиент определяет направление атаки, и его
обработка является ключевым фактором эффективности и переносимости.

Все классы наследуются от `GradientComputer` из `base.py`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Optional, List

from .base import GradientComputer, Loss


class StandardGradient(GradientComputer):
    """
    Базовый метод: вычисляет градиент функции потерь по входу.
    """

    def compute(
            self,
            surrogate_model: nn.Module,
            images: torch.Tensor,
            labels: torch.Tensor,
            loss_fn: Loss,
            all_models: List[nn.Module] = None
    ) -> torch.Tensor:
        images.requires_grad = True
        logits = surrogate_model(images)
        loss = loss_fn(logits, labels)
        surrogate_model.zero_grad()
        loss.backward()
        return images.grad.data


class MomentumGradient(GradientComputer):
    """
    Вычисляет градиент с использованием момента (Momentum Iterative Method).
    Стабилизирует направление атаки и улучшает переносимость.
    """

    def __init__(self, decay_factor: float = 1.0):
        super().__init__()
        self.decay_factor = decay_factor
        self.previous_grad: Optional[torch.Tensor] = None

    def compute(
            self,
            surrogate_model: nn.Module,
            images: torch.Tensor,
            labels: torch.Tensor,
            loss_fn: Loss,
            all_models: List[nn.Module] = None
    ) -> torch.Tensor:
        images.requires_grad = True
        logits = surrogate_model(images)
        loss = loss_fn(logits, labels)
        surrogate_model.zero_grad()
        loss.backward()

        grad = images.grad.data

        grad_norm = torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
        grad = grad / (grad_norm + 1e-12)

        if self.previous_grad is None:
            self.previous_grad = torch.zeros_like(grad)

        if self.previous_grad is None or self.previous_grad.shape != grad.shape or self.previous_grad.device != grad.device:
            self.previous_grad = torch.zeros_like(grad)

        self.previous_grad = self.decay_factor * self.previous_grad + grad

        return self.previous_grad

    def reset(self):
        """Сбрасывает накопленный градиент."""
        self.previous_grad = None


class AdversarialGradientSmoothing(GradientComputer):
    """
    Сглаживание градиента путем усреднения по нескольким зашумленным
    копиям входного изображения. Повышает робастность градиента.
    """

    def __init__(self, num_samples: int = 5, noise_stddev: float = 0.1):
        super().__init__()
        self.num_samples = num_samples
        self.noise_stddev = noise_stddev

    def compute(
            self,
            surrogate_model: nn.Module,
            images: torch.Tensor,
            labels: torch.Tensor,
            loss_fn: Loss,
            all_models: List[nn.Module] = None
    ) -> torch.Tensor:
        if self.num_samples <= 1 or self.noise_stddev == 0:
            return StandardGradient().compute(surrogate_model, images, labels, loss_fn)

        cumulative_grad = torch.zeros_like(images)

        for i in range(self.num_samples):
            noise = torch.randn_like(images) * self.noise_stddev

            noisy_images = (images.detach() + noise).requires_grad_(True)

            logits = surrogate_model(noisy_images)
            loss = loss_fn(logits, labels)

            grad, = torch.autograd.grad(
                outputs=loss,
                inputs=noisy_images,
                only_inputs=True
            )

            cumulative_grad += grad

        avg_grad = cumulative_grad / self.num_samples

        return avg_grad


class InputDiversityGradient(GradientComputer):
    """
    Применяет случайные трансформации (resize, padding) к входу
    перед вычислением градиента для повышения переносимости.
    """

    def __init__(self, prob: float = 0.7, resize_factor_min: float = 0.8, resize_factor_max: float = 1.0):
        super().__init__()
        self.prob = prob
        self.resize_factor_min = resize_factor_min
        self.resize_factor_max = resize_factor_max

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.prob:
            return x

        img_size = x.shape[-1]
        resize_factor = torch.rand(1).item() * (
                    self.resize_factor_max - self.resize_factor_min) + self.resize_factor_min
        new_size = int(img_size * resize_factor)

        resized_x = F.interpolate(x, size=new_size, mode='bilinear', align_corners=False)

        delta = img_size - new_size
        padding_top = torch.randint(0, delta + 1, (1,)).item()
        padding_bottom = delta - padding_top
        padding_left = torch.randint(0, delta + 1, (1,)).item()
        padding_right = delta - padding_left

        return F.pad(resized_x, (padding_left, padding_right, padding_top, padding_bottom), mode='constant', value=0)

    def compute(
            self,
            surrogate_model: nn.Module,
            images: torch.Tensor,
            labels: torch.Tensor,
            loss_fn: Loss,
            all_models: List[nn.Module] = None
    ) -> torch.Tensor:
        images_clone = images.clone().detach().requires_grad_(True)
        transformed_images = self.transform(images_clone)

        logits = surrogate_model(transformed_images)
        loss = loss_fn(logits, labels)
        grad, = torch.autograd.grad(loss, images_clone)
        return grad


class TranslationInvariantGradient(GradientComputer):
    """
    Усредняет градиенты по сдвинутым версиям изображения, чтобы
    сделать атаку инвариантной к сдвигам. Реализовано через свертку.
    """

    def __init__(self, kernel_size: int = 5, sigma: float = 1.0):
        super().__init__()
        self.kernel = self.get_gaussian_kernel(kernel_size, sigma)

    @staticmethod
    def get_gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
        """Создает Гауссово ядро для свертки."""
        coords = torch.arange(kernel_size, dtype=torch.float32)
        coords -= kernel_size // 2
        g = coords.pow(2)
        g = (-g / (2 * sigma ** 2)).exp()
        g /= g.sum()
        kernel = g.outer(g)
        return kernel.unsqueeze(0).unsqueeze(0)

    def compute(
            self,
            surrogate_model: nn.Module,
            images: torch.Tensor,
            labels: torch.Tensor,
            loss_fn: Loss,
            all_models: List[nn.Module] = None
    ) -> torch.Tensor:
        images.requires_grad = True
        logits = surrogate_model(images)
        loss = loss_fn(logits, labels)
        surrogate_model.zero_grad()
        loss.backward()

        grad = images.grad.data

        kernel = self.kernel.to(grad.device)
        num_channels = grad.shape[1]

        depthwise_kernel = kernel.repeat(num_channels, 1, 1, 1)

        smoothed_grad = F.conv2d(grad, depthwise_kernel, groups=num_channels, padding='same')

        return smoothed_grad


class EnsembleGradient(GradientComputer):
    """
    Вычисляет градиент как среднее градиентов от ансамбля моделей
    для повышения переносимости атаки.
    """
    def compute(
            self,
            surrogate_model: nn.Module,
            images: torch.Tensor,
            labels: torch.Tensor,
            loss_fn: Loss,
            all_models: List[nn.Module] = None
    ) -> torch.Tensor:

        if not all_models:
            # Фоллбэк на стандартный градиент, если доступна только одна модель
            return StandardGradient().compute(surrogate_model, images, labels, loss_fn, all_models)

        cumulative_grad = torch.zeros_like(images, requires_grad=False)

        for model in all_models:
            # Важно создавать копию images, чтобы градиенты не смешивались
            images_clone = images.clone().detach().requires_grad_(True)

            logits = model(images_clone)
            loss = loss_fn(logits, labels)

            model.zero_grad()  # Обнуляем градиенты конкретной модели

            # Вычисляем градиент только по `images_clone`
            grad, = torch.autograd.grad(loss, images_clone, only_inputs=True)

            cumulative_grad += grad.detach()

        # Усредняем градиент
        avg_grad = cumulative_grad / len(all_models)
        return avg_grad


class SkipReLUFunction(Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        # Просто передаем градиент дальше, как если бы это был Identity
        return grad_output


class SGM(GradientComputer):
    """
    Реализует Skip Gradient Method (SGM) путем замены ReLU на кастомную
    функцию с "прямым" градиентом во время обратного прохода.
    """
    def __init__(self):
        super().__init__()
        self.original_relus = {}

    def _replace_relu(self, module):
        """Рекурсивно заменяет все ReLU в модели."""
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU):
                # Сохраняем оригинальный ReLU, чтобы потом восстановить
                self.original_relus[child] = child.forward
                # Заменяем
                child.forward = lambda x: SkipReLUFunction.apply(x)
            else:
                self._replace_relu(child)

    def _restore_relu(self, module):
        """Восстанавливает оригинальные ReLU."""
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU) and child in self.original_relus:
                child.forward = self.original_relus[child]
            else:
                self._restore_relu(child)
        self.original_relus = {}

    def compute(
            self,
            surrogate_model: nn.Module,
            images: torch.Tensor,
            labels: torch.Tensor,
            loss_fn: Loss,
            all_models: List[nn.Module] = None
    ) -> torch.Tensor:

        # Оборачиваем вычисление в try-finally для гарантии восстановления
        try:
            # 1. Заменяем все ReLU в модели на нашу "пропускающую" версию
            self._replace_relu(surrogate_model)

            # 2. Вычисляем градиент как обычно
            images.requires_grad = True
            logits = surrogate_model(images)
            loss = loss_fn(logits, labels)
            surrogate_model.zero_grad()
            loss.backward()
            grad = images.grad.data

        finally:
            # 3. Восстанавливаем оригинальные ReLU, чтобы не сломать модель
            self._restore_relu(surrogate_model)

        return grad


class TAP(GradientComputer):
    """
    Вычисляет градиент на основе Transferable Adversarial Perturbations (TAP).
    Минимизирует CE loss на суррогатной модели и KL-дивергенцию между
    логитами суррогатной модели и других моделей в ансамбле.
    """
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.kl_loss_fn = nn.KLDivLoss(reduction='batchmean')

    def compute(
            self,
            surrogate_model: nn.Module,
            images: torch.Tensor,
            labels: torch.Tensor,
            loss_fn: Loss,
            all_models: List[nn.Module] = None
    ) -> torch.Tensor:

        if not all_models or len(all_models) < 2:
            # Фоллбэк, если нет ансамбля для сравнения
            return StandardGradient().compute(surrogate_model, images, labels, loss_fn, all_models)

        images.requires_grad = True

        # 1. Вычисляем CE loss на суррогатной модели
        surrogate_logits = surrogate_model(images)
        ce_loss = loss_fn(surrogate_logits, labels)

        # 2. Вычисляем loss на основе расхождения логитов
        kl_loss = torch.tensor(0.0, device=images.device)
        surrogate_log_softmax = F.log_softmax(surrogate_logits, dim=1)

        # Получаем список "других" моделей
        other_models = [m for m in all_models if m is not surrogate_model]

        with torch.no_grad():  # Градиент от других моделей нам не нужен
            for other_model in other_models:
                other_logits = other_model(images)
                other_softmax = F.softmax(other_logits, dim=1)
                kl_loss += self.kl_loss_fn(surrogate_log_softmax, other_softmax)

        if other_models:
            kl_loss /= len(other_models)

        # 3. Комбинируем потери
        total_loss = ce_loss + self.alpha * kl_loss

        # 4. Вычисляем градиент от суммарной потери
        surrogate_model.zero_grad()
        total_loss.backward()

        return images.grad.data