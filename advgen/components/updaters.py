from .base import UpdateRule
import torch
from typing import Optional


class StandardUpdate(UpdateRule):
    """
    Самое простое правило обновления: делает шаг прямо по направлению
    предоставленного градиента без какой-либо нормализации или модификации.
    """
    def update(self, images: torch.Tensor, grad: torch.Tensor, step_size: float) -> torch.Tensor:
        return images.detach() + step_size * grad


class SignUpdate(UpdateRule):
    """
    Классическое обновление, использующее знак градиента.
    Является стандартным для L-infinity атак (например, PGD, FGSM).
    Гарантирует, что каждый шаг имеет одинаковую L-infinity величину.
    """
    def update(self, images: torch.Tensor, grad: torch.Tensor, step_size: float) -> torch.Tensor:
        return images.detach() + step_size * torch.sign(grad)


class L2Update(UpdateRule):
    """
    Обновление для L2-атак, использует L2-нормализованный градиент.
    Гарантирует, что каждый шаг имеет одинаковую евклидову длину (L2-норму).
    """
    def update(self, images: torch.Tensor, grad: torch.Tensor, step_size: float) -> torch.Tensor:
        batch_size = grad.shape[0]
        flat_grad = grad.view(batch_size, -1)
        norms = torch.linalg.norm(flat_grad, ord=2, dim=1)
        normalized_grad = grad / torch.clamp(norms.view(batch_size, 1, 1, 1), min=1e-12)
        return images.detach() + step_size * normalized_grad


class AdamUpdate(UpdateRule):
    """
    Использует логику оптимизатора Adam для шага атаки.
    Адаптивно подбирает размер шага для каждого параметра (пикселя),
    используя оценки первого (момент) и второго (скользящее среднее квадрата)
    моментов градиента. Может быть эффективен для сложных ландшафтов потерь.
    """
    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        super().__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None
        self.t: int = 0

    def update(self, images: torch.Tensor, grad: torch.Tensor, step_size: float) -> torch.Tensor:
        self.t += 1

        if self.m is None or self.m.shape != grad.shape:
            self.m = torch.zeros_like(grad)
            self.v = torch.zeros_like(grad)

        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad.pow(2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        update_step = step_size * m_hat / (torch.sqrt(v_hat) + self.eps)

        return images.detach() + update_step

    def reset(self):
        """Сбрасывает внутреннее состояние (моменты и счетчик шагов)."""
        self.m = None
        self.v = None
        self.t = 0


class L1MedianUpdate(UpdateRule):
    """
    Правило обновления, адаптированное для L1-атак.
    Использует знак градиента, но нормализует его по медиане абсолютных
    значений, что делает обновление более стабильным для разреженных атак.
    Используется, например, в атаке EAD (Elastic-Net Attack).
    """

    def update(self, images: torch.Tensor, grad: torch.Tensor, step_size: float) -> torch.Tensor:
        batch_size = grad.shape[0]
        flat_grad = grad.view(batch_size, -1)

        median, _ = torch.median(torch.abs(flat_grad), dim=1, keepdim=True)

        grad_sign = torch.sign(grad)
        normalized_grad = grad_sign / torch.clamp(median.view(batch_size, 1, 1, 1), min=1e-12)

        return images.detach() + step_size * normalized_grad