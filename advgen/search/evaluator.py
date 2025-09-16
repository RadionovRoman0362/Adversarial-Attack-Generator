"""
Этот модуль содержит функцию-оценщик (Evaluator), которая является
ядром экспериментального цикла.

Ее задача - взять одну полную конфигурацию атаки, запустить ее на всем
тестовом датасете и агрегировать результаты, чтобы вернуть итоговые метрики.
"""

import copy
import logging
from typing import Dict, Any, List

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

    if "warmup_steps_rel" in scheduler_params:
        total_steps = processed_config.get("steps")
        rel_warmup = scheduler_params.pop("warmup_steps_rel")
        abs_warmup = int(rel_warmup * total_steps)
        scheduler_params["warmup_steps"] = abs_warmup
        logger.debug(f"Конвертирован warmup_steps_rel {rel_warmup} -> {abs_warmup}")

    if "cycle_steps_rel" in scheduler_params:
        total_steps = processed_config.get("steps")
        rel_cycle = scheduler_params.pop("cycle_steps_rel")
        abs_cycle = int(rel_cycle * total_steps)
        scheduler_params["cycle_steps"] = abs_cycle
        logger.debug(f"Конвертирован cycle_steps_rel {rel_cycle} -> {abs_cycle}")

    return processed_config


def evaluate_config(
        attack_config: Dict[str, Any],
        model_wrappers: List[ModelWrapper],
        dataloader: DataLoader,
        device: torch.device,
        num_examples_to_return: int = 0,
        progress_info: str = ""
) -> tuple[dict[str, Any], list | None]:
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
    :param model_wrappers: Обертки над атакуемыми моделями.
    :param dataloader: DataLoader с тестовыми данными.
    :param device: Устройство, на котором будут производиться вычисления ('cpu' или 'cuda').
    :param num_examples_to_return: Количество изображений, которое надо сохранить.
    :param progress_info: Номер текущего индивида для эволюционного алгоритма.
    :return: Словарь с итоговыми метриками по всему датасету.
    """
    if not model_wrappers:
        raise ValueError("Список model_wrappers не может быть пустым.")

    surrogate_model = model_wrappers[0]
    num_models = len(model_wrappers)

    processed_attack_config = _preprocess_config(attack_config)

    try:
        attack_runner = AttackRunner(processed_attack_config, model_wrappers)
        progress_prefix = f"{progress_info} | " if progress_info else ""
        logger.info(f"{progress_prefix}Оценка конфигурации: {attack_runner}")
    except (ValueError, TypeError) as e:
        logger.error(f"Не удалось создать AttackRunner с данной конфигурацией. Ошибка: {e}")
        logger.debug(f"Проблемная конфигурация: {processed_attack_config}")
        return {
            "attack_success_rate": -1.0,
            "robust_accuracy": -1.0,
            "error": str(e)
        }, None

    total_initially_correct = 0
    total_successful_attacks = 0
    cumulative_linf = 0.0
    cumulative_l2 = 0.0
    cumulative_l1 = 0.0

    examples_to_return = []

    pbar_desc = "Оценка атаки"
    if progress_info:
        pbar_desc = f"{progress_info}"

    pbar = tqdm(dataloader, desc=pbar_desc, leave=False, dynamic_ncols=True)

    for i, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        with autocast('cuda', dtype=torch.float16):
            adv_images = attack_runner.attack(surrogate_model, images, labels)
        adv_images = adv_images.float()

        for model_wrapper in model_wrappers:
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

            if i == 0 and num_examples_to_return > 0:
                clean_logits = model_wrapper(images)
                clean_preds = torch.argmax(clean_logits, dim=1)
                adv_logits = model_wrapper(adv_images)
                adv_preds = torch.argmax(adv_logits, dim=1)

                for j in range(min(num_examples_to_return, len(images))):
                    examples_to_return.append({
                        "original_image": images[j].cpu(),
                        "adv_image": adv_images[j].cpu(),
                        "label": labels[j].cpu().item(),
                        "clean_pred": clean_preds[j].cpu().item(),
                        "adv_pred": adv_preds[j].cpu().item()
                    })

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

    logger.info(f"{progress_prefix}Результаты оценки (по {num_models} моделям): "
                f"Avg ASR={results['attack_success_rate']:.4f}, "
                f"RobustAcc={results['robust_accuracy']:.4f}, "
                f"Avg L2={results['avg_l2_norm']:.4f}")

    return results, examples_to_return if examples_to_return else None