"""
Этот модуль содержит реализации методов вычисления и модификации градиента,
построенные на основе паттерна "Декоратор".

Архитектура:
1.  **Базовый компонент (`StandardGradient`):**
    Это "конечная точка" любой цепочки. Он непосредственно взаимодействует с PyTorch
    для вычисления градиента функции потерь по входу (`loss.backward()`).

2.  **Компоненты-декораторы:**
    Все остальные классы являются декораторами. Каждый из них принимает в конструкторе
    другой `GradientComputer` (аргумент `wrapped_computer`) и "оборачивает" его.

    В своем методе `compute` декоратор:
    a) Может изменить входные данные ПЕРЕД передачей их вложенному компоненту
       (например, `AdversarialGradientSmoothing` добавляет шум).
    b) Вызывает `self.wrapped_computer.compute(...)` для получения базового градиента.
    c) Может модифицировать полученный градиент ПОСЛЕ его вычисления
       (например, `MomentumGradient` добавляет момент, а `TranslationInvariantGradient`
       применяет свертку).

Эта архитектура позволяет создавать сложные, композитные методы вычисления градиента,
гибко комбинируя их в конфигурации, например:
`TI(Momentum(StandardGradient))`
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Optional, List

from .base import GradientComputer, Loss


class StandardGradient(GradientComputer):
    """
    Терминальный компонент: вычисляет "чистый" градиент функции потерь по входу.
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
        grad = images.grad.data
        images.grad = None
        images.requires_grad_(False)
        return grad


class InputDiversityGradient(GradientComputer):
    """
    Терминальный компонент: применяет случайные трансформации к входу
    и вычисляет градиент по отношению к оригинальному входу.
    Повышает переносимость.
    """
    def __init__(
            self,
            prob: float = 0.7,
            resize_factor_min: float = 0.8,
            resize_factor_max: float = 1.0
    ):
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

        surrogate_model.zero_grad()
        grad, = torch.autograd.grad(loss, images_clone)

        return grad


class TAP(GradientComputer):
    """
    Терминальный компонент: вычисляет градиент на основе комбинированной
    функции потерь (CE + KL-дивергенция).
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
            return StandardGradient().compute(surrogate_model, images, labels, loss_fn, all_models)

        images.requires_grad = True

        surrogate_logits = surrogate_model(images)
        ce_loss = loss_fn(surrogate_logits, labels)

        kl_loss = torch.tensor(0.0, device=images.device)
        surrogate_log_softmax = F.log_softmax(surrogate_logits, dim=1)
        other_models = [m for m in all_models if m is not surrogate_model]

        with torch.no_grad():
            for other_model in other_models:
                other_logits = other_model(images)
                other_softmax = F.softmax(other_logits, dim=1)
                kl_loss += self.kl_loss_fn(surrogate_log_softmax, other_softmax)

        if other_models:
            kl_loss /= len(other_models)

        total_loss = ce_loss + self.alpha * kl_loss

        surrogate_model.zero_grad()
        total_loss.backward()

        grad = images.grad.data
        images.grad = None
        images.requires_grad_(False)

        return grad


class MomentumGradient(GradientComputer):
    """
    Декоратор: добавляет инерцию (момент) к вычисленному градиенту.
    Стабилизирует направление атаки.
    Оборачивает: любой `GradientComputer`.
    Действие: пост-обработка градиента.
    """
    def __init__(self, wrapped_computer: GradientComputer, decay_factor: float = 1.0):
        super().__init__()
        self.wrapped_computer = wrapped_computer
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
        raw_grad = self.wrapped_computer.compute(
            surrogate_model, images, labels, loss_fn, all_models
        )

        grad_norm = torch.mean(torch.abs(raw_grad), dim=(1, 2, 3), keepdim=True)
        normalized_grad = raw_grad / (grad_norm + 1e-12)

        if self.previous_grad is None or self.previous_grad.shape != normalized_grad.shape or self.previous_grad.device != normalized_grad.device:
            self.previous_grad = torch.zeros_like(normalized_grad)

        self.previous_grad = self.decay_factor * self.previous_grad + normalized_grad

        return self.previous_grad

    def reset(self):
        """Сбрасывает накопленный градиент."""
        self.previous_grad = None
        self.wrapped_computer.reset()


class AdversarialGradientSmoothing(GradientComputer):
    """
    Декоратор: усредняет градиенты по нескольким зашумленным копиям входа.
    Повышает робастность градиента.
    Оборачивает: любой `GradientComputer`.
    Действие: модификация процесса вычисления (многократный вызов).
    """
    def __init__(self, wrapped_computer: GradientComputer, num_samples: int = 5, noise_stddev: float = 0.1):
        super().__init__()
        self.wrapped_computer = wrapped_computer
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
            return self.wrapped_computer.compute(surrogate_model, images, labels, loss_fn, all_models)

        cumulative_grad = torch.zeros_like(images)
        original_images_detached = images.detach()

        for _ in range(self.num_samples):
            noise = torch.randn_like(images) * self.noise_stddev
            noisy_images = original_images_detached + noise

            grad_sample = self.wrapped_computer.compute(
                surrogate_model, noisy_images, labels, loss_fn, all_models
            )
            cumulative_grad += grad_sample

        return cumulative_grad / self.num_samples

    def reset(self):
        self.wrapped_computer.reset()


class TranslationInvariantGradient(GradientComputer):
    """
    Декоратор: сглаживает вычисленный градиент с помощью Гауссовой свертки,
    делая атаку инвариантной к сдвигам.
    Оборачивает: любой `GradientComputer`.
    Действие: пост-обработка градиента.
    """
    def __init__(self, wrapped_computer: GradientComputer, kernel_size: int = 5, sigma: float = 1.0):
        super().__init__()
        self.wrapped_computer = wrapped_computer
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
        grad = self.wrapped_computer.compute(surrogate_model, images, labels, loss_fn, all_models)

        kernel = self.kernel.to(grad.device)
        num_channels = grad.shape[1]
        depthwise_kernel = kernel.repeat(num_channels, 1, 1, 1)
        smoothed_grad = F.conv2d(grad, depthwise_kernel, groups=num_channels, padding='same')

        return smoothed_grad

    def reset(self):
        self.wrapped_computer.reset()


class EnsembleGradient(GradientComputer):
    """
    Декоратор: усредняет градиенты от ансамбля моделей.
    Оборачивает: любой `GradientComputer`, чтобы определить, КАКОЙ градиент
    (например, стандартный или с моментом) считать для каждой модели.
    Действие: модификация процесса (вызов на нескольких моделях).
    """
    def __init__(self, wrapped_computer: GradientComputer):
        super().__init__()
        self.wrapped_computer = wrapped_computer

    def compute(
            self,
            surrogate_model: nn.Module,
            images: torch.Tensor,
            labels: torch.Tensor,
            loss_fn: Loss,
            all_models: List[nn.Module] = None
    ) -> torch.Tensor:
        if not all_models or len(all_models) <= 1:
            return self.wrapped_computer.compute(surrogate_model, images, labels, loss_fn, [surrogate_model])

        cumulative_grad = torch.zeros_like(images, requires_grad=False)

        for model in all_models:
            grad_sample = self.wrapped_computer.compute(
                model, images, labels, loss_fn, all_models=[model]
            )
            cumulative_grad += grad_sample.detach()

        return cumulative_grad / len(all_models)

    def reset(self):
        self.wrapped_computer.reset()


class SkipReLUFunction(Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class SGM(GradientComputer):
    """
    Декоратор (Skip Gradient Method): модифицирует модель, заменяя ReLU,
    перед вычислением градиента вложенным компонентом.
    Оборачивает: любой `GradientComputer`.
    Действие: модификация модели (setup/teardown).
    """
    def __init__(self, wrapped_computer: GradientComputer):
        super().__init__()
        self.wrapped_computer = wrapped_computer
        self.original_relus = {}

    def _replace_relu(self, module):
        """Рекурсивно заменяет все ReLU в модели."""
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU):
                self.original_relus[child] = child.forward
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
        target_model = surrogate_model
        try:
            self._replace_relu(target_model)
            grad = self.wrapped_computer.compute(
                target_model, images, labels, loss_fn, all_models
            )
        finally:
            self._restore_relu(target_model)

        return grad

    def reset(self):
        self.wrapped_computer.reset()