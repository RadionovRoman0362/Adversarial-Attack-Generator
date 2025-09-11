"""
Этот модуль содержит конкретные реализации функций потерь для нецелевых атак.

Каждая функция потерь определяет стратегию, которую использует атака для
"обмана" модели. Все классы наследуются от `Loss` из `base.py`.

По соглашению, все функции возвращают значение, которое необходимо **максимизировать**
в ходе атаки.
"""

import torch
import torch.nn.functional as F

from .base import Loss


class CrossEntropyLoss(Loss):
    """
    Стандартная кросс-энтропия. Атака максимизирует эту потерю, чтобы
    снизить уверенность модели в правильном классе.
    """

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет кросс-энтропию и возвращает ее среднее значение по батчу.
        """
        return F.cross_entropy(logits, labels, reduction='mean')


class UntargetedCWLoss(Loss):
    """
    Функция потерь Карлини-Вагнера для нецелевой атаки.
    Атака максимизирует разницу между логитом правильного класса и
    максимальным логитом среди всех неправильных классов.
    Формула для максимизации: max_{i!=t}(Z_i) - Z_t, где Z - логиты, t - верная метка.
    """

    def __init__(self, kappa: float = 0):
        super().__init__()
        self.kappa = kappa

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет CW-loss для нецелевой атаки.
        """
        one_hot_labels = F.one_hot(labels, num_classes=logits.shape[1])
        correct_logits = (logits * one_hot_labels).sum(dim=1)

        other_logits = torch.where(one_hot_labels.bool(), -torch.inf, logits)
        max_other_logits = torch.max(other_logits, dim=1)[0]

        loss = torch.clamp(max_other_logits - correct_logits, min=-self.kappa)
        return loss.mean()


class DifferenceOfLogitsRatioLoss(Loss):
    """
    Функция потерь DLR (Difference of Logits Ratio) из статьи Auto-Attack.
    Это более робастная версия margin-based loss.
    Мы минимизируем (Z_t - max_{i!=t}(Z_i)) / (Z_pi(1) - Z_pi(3)),
    что эквивалентно максимизации обратной величины. Для простоты, мы
    максимизируем -(Z_t - max_{i!=t}(Z_i)), нормированное на знаменатель.
    """

    def __init__(self, k: int = 3):
        super().__init__()
        self.k = k

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет DLR-loss.
        """
        one_hot_labels = F.one_hot(labels, num_classes=logits.shape[1])
        correct_logits = (logits * one_hot_labels).sum(dim=1)

        other_logits = torch.where(one_hot_labels.bool(), -torch.inf, logits)
        max_other_logits = torch.max(other_logits, dim=1)[0]

        numerator = max_other_logits - correct_logits

        sorted_logits, _ = torch.sort(logits, dim=1, descending=True)

        actual_k = min(self.k, sorted_logits.shape[1] - 1)

        top_1_logits = sorted_logits[:, 0]
        top_k_logits = sorted_logits[:, actual_k]

        denominator = top_1_logits - top_k_logits

        loss = numerator / torch.clamp(denominator, min=1e-12)

        return loss.mean()


class PopperLoss(Loss):
    """
    Функция потерь, максимизирующая KL-дивергенцию между выходом модели
    и равномерным распределением по всем *неправильным* классам.
    Это заставляет модель стать максимально "неуверенной".
    """

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет Popper loss.
        """
        num_classes = logits.shape[1]
        if num_classes <= 1:
            return torch.tensor(0.0, device=logits.device)

        target_probs = torch.full_like(logits, 1.0 / (num_classes - 1))
        target_probs.scatter_(1, labels.unsqueeze(1), 0)

        log_probs = F.log_softmax(logits, dim=1)

        kl_div = F.kl_div(log_probs, target_probs, reduction='none').sum(dim=1)

        return kl_div.mean()