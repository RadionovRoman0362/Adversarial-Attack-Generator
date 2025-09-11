"""
Этот модуль содержит функцию-оценщик (Evaluator), которая является
ядром экспериментального цикла.

Ее задача - взять одну полную конфигурацию атаки, запустить ее на всем
тестовом датасете и агрегировать результаты, чтобы вернуть итоговые метрики.
"""

import copy
import logging
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm

from ..core.attack_runner import AttackRunner
from ..core.model_wrapper import ModelWrapper
from ..utils.metrics import calculate_attack_metrics, get_attack_success_rate, get_robust_accuracy

logger = logging.getLogger(__name__)


def _preprocess_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Выполняет предварительную обработку конфигурации атаки.
    В частности, конвертирует относительные "вехи" в абсолютные.
    """
    processed_config = copy.deepcopy(config)

    scheduler_config = processed_config.get("scheduler", {})
    scheduler_params = scheduler_config.get("params", {})

    if "milestones_rel" in scheduler_params:
        total_steps = processed_config.get("steps")
        if total_steps is None:
            raise ValueError("Для использования 'milestones_rel' в конфиге "
                             "должен быть указан параметр 'steps'.")

        rel_milestones = scheduler_params.pop("milestones_rel")
        abs_milestones = [int(m * total_steps) for m in rel_milestones]
        scheduler_params["milestones"] = abs_milestones
        logger.debug(f"Конвертированы относительные вехи {rel_milestones} "
                     f"в абсолютные {abs_milestones} для {total_steps} шагов.")

    return processed_config


def evaluate_config(
        attack_config: Dict[str, Any],
        model_wrapper: ModelWrapper,
        dataloader: DataLoader,
        device: torch.device
) -> Dict[str, Any]:
    """
    Оценивает одну полную конфигурацию состязательной атаки.

    Процесс:
    1. Создает экземпляр AttackRunner с данной конфигурацией.
    2. Итерируется по всему датасету из DataLoader'а.
    3. Для каждого батча запускает атаку.
    4. Собирает метрики (число успешных атак, нормы возмущений) для каждого батча.
    5. Агрегирует метрики по всему датасету и вычисляет итоговые значения
       (ASR, Robust Accuracy, средние нормы).

    :param attack_config: Словарь с конфигурацией атаки, готовый для AttackRunner.
    :param model_wrapper: Обертка над атакуемой моделью.
    :param dataloader: DataLoader с тестовыми данными.
    :param device: Устройство, на котором будут производиться вычисления ('cpu' или 'cuda').
    :return: Словарь с итоговыми метриками по всему датасету.
    """
    processed_attack_config = _preprocess_config(attack_config)

    try:
        attack_runner = AttackRunner(processed_attack_config)
        logger.info(f"Оценка конфигурации: {attack_runner}")
    except (ValueError, TypeError) as e:
        logger.error(f"Не удалось создать AttackRunner с данной конфигурацией. Ошибка: {e}")
        logger.debug(f"Проблемная конфигурация: {processed_attack_config}")
        return {
            "attack_success_rate": -1.0,
            "robust_accuracy": -1.0,
            "error": str(e)
        }

    total_initially_correct = 0
    total_successful_attacks = 0
    cumulative_linf = 0.0
    cumulative_l2 = 0.0
    cumulative_l1 = 0.0

    pbar = tqdm(dataloader, desc="Оценка атаки", leave=False, dynamic_ncols=True)

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        with autocast('cuda', dtype=torch.float16):
            adv_images = attack_runner.attack(model_wrapper, images, labels)

        adv_images = adv_images.float()

        (
            init_correct,
            success_count,
            batch_linf,
            batch_l2,
            batch_l1
        ) = calculate_attack_metrics(model_wrapper, adv_images, images, labels)

        total_initially_correct += init_correct
        total_successful_attacks += success_count
        cumulative_linf += batch_linf
        cumulative_l2 += batch_l2
        cumulative_l1 += batch_l1

        current_asr = get_attack_success_rate(total_successful_attacks, total_initially_correct)
        pbar.set_postfix(CurrentASR=f"{current_asr:.3f}")

    attack_success_rate = get_attack_success_rate(total_successful_attacks, total_initially_correct)
    robust_accuracy = get_robust_accuracy(total_successful_attacks, total_initially_correct)

    avg_linf = cumulative_linf / total_successful_attacks if total_successful_attacks > 0 else 0.0
    avg_l2 = cumulative_l2 / total_successful_attacks if total_successful_attacks > 0 else 0.0
    avg_l1 = cumulative_l1 / total_successful_attacks if total_successful_attacks > 0 else 0.0

    results = {
        "attack_success_rate": attack_success_rate,
        "robust_accuracy": robust_accuracy,
        "avg_linf_norm": avg_linf,
        "avg_l2_norm": avg_l2,
        "avg_l1_norm": avg_l1,
        "total_successful_attacks": total_successful_attacks,
        "total_evaluated": total_initially_correct,
        "processed_config": processed_attack_config
    }

    logger.info(f"Результаты оценки: ASR={results['attack_success_rate']:.4f}, "
                f"RobustAcc={results['robust_accuracy']:.4f}, "
                f"Avg L2={results['avg_l2_norm']:.4f}")

    return results