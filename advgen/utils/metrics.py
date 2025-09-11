"""
Этот модуль содержит функции для вычисления метрик, используемых
для оценки эффективности состязательных атак.
"""
from typing import Tuple
import torch
import torch.nn as nn


@torch.no_grad()
def calculate_attack_metrics(
        model_wrapper: nn.Module,
        adv_images: torch.Tensor,
        original_images: torch.Tensor,
        labels: torch.Tensor
) -> Tuple[int, int, float, float, float]:
    """
    Вычисляет комплексные метрики для одного батча.
    Атака считается на изображениях, которые модель изначально классифицирует верно.

    :param model_wrapper: Обертка над моделью для получения логитов.
    :param adv_images: Батч состязательных изображений.
    :param original_images: Батч оригинальных изображений.
    :param labels: Истинные метки для изображений.
    :return: Кортеж, содержащий:
             - initially_correct_count (int): Кол-во изначально верных предсказаний.
             - successful_attacks_count (int): Кол-во успешных атак в батче.
             - total_linf (float): Сумма L-inf норм для успешных атак.
             - total_l2 (float): Сумма L2 норм для успешных атак.
             - total_l1 (float): Сумма L1 норм для успешных атак.
    """
    clean_logits = model_wrapper(original_images)
    clean_preds = torch.argmax(clean_logits, dim=1)
    initially_correct_mask = (clean_preds == labels)
    initially_correct_count = initially_correct_mask.sum().item()

    if initially_correct_count == 0:
        return 0, 0, 0.0, 0.0, 0.0

    adv_logits_on_correct = model_wrapper(adv_images[initially_correct_mask])
    adv_preds_on_correct = torch.argmax(adv_logits_on_correct, dim=1)
    original_labels_on_correct = labels[initially_correct_mask]

    successful_mask = (adv_preds_on_correct != original_labels_on_correct)
    successful_attacks_count = successful_mask.sum().item()

    if successful_attacks_count > 0:
        perturbations = (adv_images - original_images)[initially_correct_mask][successful_mask]

        total_linf = perturbations.abs().view(successful_attacks_count, -1).max(dim=1)[0].sum().item()
        total_l2 = torch.linalg.norm(perturbations.view(successful_attacks_count, -1), ord=2, dim=1).sum().item()
        total_l1 = torch.linalg.norm(perturbations.view(successful_attacks_count, -1), ord=1, dim=1).sum().item()
    else:
        total_linf, total_l2, total_l1 = 0.0, 0.0, 0.0

    return initially_correct_count, successful_attacks_count, total_linf, total_l2, total_l1


def get_attack_success_rate(
        successful_attacks_total: int,
        total_samples: int
) -> float:
    """
    Вычисляет итоговый процент успешных атак (Attack Success Rate, ASR).

    :param successful_attacks_total: Общее число успешных атак по всему датасету.
    :param total_samples: Общее число обработанных примеров.
    :return: Процент успешных атак (от 0.0 до 1.0).
    """
    if total_samples == 0:
        return 0.0
    return successful_attacks_total / total_samples


def get_robust_accuracy(
        successful_attacks_total: int,
        total_samples: int
) -> float:
    """
    Вычисляет робастную точность (Robust Accuracy).
    Это доля примеров, на которых атака НЕ удалась.

    :param successful_attacks_total: Общее число успешных атак.
    :param total_samples: Общее число обработанных примеров.
    :return: Робастная точность (от 0.0 до 1.0).
    """
    if total_samples == 0:
        return 0.0
    return 1.0 - (successful_attacks_total / total_samples)