import torch
import torch.nn as nn
import torch.nn.functional as F

class TRADESLoss(nn.Module):
    """
    Реализует функцию потерь TRADES, которая состоит из двух частей:
    1. Потери на чистых данных (стандартная кросс-энтропия).
    2. Потери на робастность (KL-дивергенция между выходами на чистых и состязательных данных).
    """
    def __init__(self, beta: float = 6.0):
        super(TRADESLoss, self).__init__()
        if beta < 0:
            raise ValueError("Параметр beta для TRADES должен быть неотрицательным.")
        self.beta = beta
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_kl = nn.KLDivLoss(reduction='batchmean')

    def forward(self, logits_clean: torch.Tensor, logits_adv: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет общую потерю TRADES.

        :param logits_clean: Логиты модели на чистых данных.
        :param logits_adv: Логиты модели на состязательных данных.
        :param labels: Истинные метки.
        :return: Скалярный тензор с итоговой потерей.
        """
        loss_ce = self.criterion_ce(logits_clean, labels)

        log_probs_adv = F.log_softmax(logits_adv, dim=1)
        probs_clean = F.softmax(logits_clean, dim=1)

        loss_kl = self.criterion_kl(log_probs_adv, probs_clean.detach())

        total_loss = loss_ce + self.beta * loss_kl
        return total_loss