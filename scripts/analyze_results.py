"""
Скрипт для комплексного анализа и визуализации результатов поиска атак.

Функционал:
1. Загружает JSON-файл с результатами, полученный от `run_search.py`.
2. (Для Optuna) Строит графики процесса оптимизации и важности параметров.
3. Оценивает лучшую найденную атаку и базовые атаки (PGD, FGSM) на одном
   и том же наборе данных для честного сравнения.
4. Строит столбчатую диаграмму, сравнивающую атаки по ASR и нормам возмущений.
5. Генерирует и сохраняет визуальные примеры лучшей атаки:
   (оригинал, разница, атакованное изображение).

Пример запуска:
> python scripts/analyze_results.py --results results/search_results_optuna_....json
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
import optuna
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from advgen.core.model_wrapper import ModelWrapper
from advgen.models import resnet_cifar
from advgen.search.evaluator import evaluate_config
from advgen.training.utils import load_checkpoint
from advgen.utils.data_loader import get_dataloader, DATASET_STATS
from advgen.utils.logging_setup import setup_logging


# --- Функции для визуализации ---

def plot_attack_examples(
        examples: List[Dict[str, Any]],
        class_names: List[str],
        save_path: str,
        num_to_show: int = 5
):
    """Визуализирует и сохраняет примеры атак."""
    num_to_show = min(num_to_show, len(examples))
    if num_to_show == 0:
        return

    fig, axes = plt.subplots(num_to_show, 3, figsize=(12, 4 * num_to_show))
    fig.suptitle("Примеры состязательных атак", fontsize=16, y=1.02)

    for i in range(num_to_show):
        example = examples[i]
        orig_img = example["original_image"].permute(1, 2, 0).numpy()
        adv_img = example["adv_image"].permute(1, 2, 0).numpy()
        diff = np.abs(adv_img - orig_img)
        diff = (diff - diff.min()) / (diff.max() - diff.min())  # Нормализация для наглядности

        label = class_names[example["label"]]
        clean_pred = class_names[example["clean_pred"]]
        adv_pred = class_names[example["adv_pred"]]

        # Оригинал
        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_title(f"Оригинал\nИстина: {label}\nПредсказание: {clean_pred}")
        axes[i, 0].axis('off')

        # Разница (возмущение)
        axes[i, 1].imshow(diff)
        axes[i, 1].set_title("Возмущение (усилено)")
        axes[i, 1].axis('off')

        # Атакованное изображение
        axes[i, 2].imshow(adv_img)
        axes[i, 2].set_title(f"Атака\nИстина: {label}\nПредсказание: {adv_pred}")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Визуальные примеры сохранены в: {save_path}")


def plot_comparison_barchart(df: pd.DataFrame, save_path: str):
    """Строит и сохраняет столбчатую диаграмму для сравнения атак."""
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")

    # Готовим данные для двойной диаграммы
    df_melted = df.melt(
        id_vars='name',
        value_vars=['attack_success_rate', 'avg_linf_norm', 'avg_l2_norm'],
        var_name='Metric',
        value_name='Value'
    )

    g = sns.catplot(
        data=df_melted, kind="bar",
        x="name", y="Value", hue="Metric",
        errorbar=None, palette="viridis", alpha=.8, height=6, aspect=1.5
    )
    g.despine(left=True)
    g.set_axis_labels("Атака", "Значение")
    g.legend.set_title("")
    plt.title("Сравнение эффективности атак", fontsize=16)

    plt.savefig(save_path)
    plt.close()
    logging.info(f"График сравнения сохранен в: {save_path}")


def get_model(config: Dict[str, Any]) -> nn.Module:
    """Фабричная функция для создания моделей на основе конфига."""
    model_name = config['name']
    num_classes = config.get('num_classes', 10)
    if model_name == 'resnet18_cifar':
        return resnet_cifar.resnet18_cifar(num_classes=num_classes, pretrained=False)
    else:
        raise NotImplementedError(f"Модель '{model_name}' не поддерживается.")


def main(results_path: str):
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info(f"--- 1. Загрузка результатов из '{results_path}' ---")
    with open(results_path, 'r', encoding='utf-8') as f:
        results_data = json.load(f)

    exp_config = results_data['experiment_config']
    best_found_config = results_data['best_performing_attack']['processed_config']

    # Создаем директорию для сохранения графиков
    analysis_dir = os.path.dirname(results_path)
    base_filename = os.path.splitext(os.path.basename(results_path))[0]
    plots_dir = os.path.join(analysis_dir, f"analysis_{base_filename}")
    os.makedirs(plots_dir, exist_ok=True)
    logger.info(f"Графики и анализ будут сохранены в: {plots_dir}")

    # --- 2. Визуализация процесса поиска (только для Optuna) ---
    if exp_config.get('search', {}).get('sampler') == 'optuna':
        logger.info("--- 2. Генерация графиков анализа Optuna ---")
        try:
            # Восстанавливаем Study из сохраненных данных
            study = optuna.create_study(direction=exp_config['search']['direction'])
            all_trials_data = results_data['all_trials']
            for trial_data in all_trials_data:
                # Пропускаем неудачные прогоны
                if trial_data.get('error'):
                    continue
                # Optuna требует, чтобы параметры были "плоскими"
                flat_params = pd.json_normalize(trial_data['processed_config'], sep='_').to_dict(orient='records')[0]
                study.add_trial(
                    optuna.create_trial(
                        value=trial_data['attack_success_rate'],
                        params=flat_params
                    )
                )

            # Сохраняем графики
            study.plot_optimization_history().write_image(os.path.join(plots_dir, "optuna_history.png"))
            study.plot_param_importances().write_image(os.path.join(plots_dir, "optuna_importances.png"))
            logger.info("Графики Optuna успешно сохранены.")
        except Exception as e:
            logger.error(f"Не удалось сгенерировать графики Optuna: {e}")

    # --- 3. Подготовка к сравнению атак ---
    logger.info("--- 3. Подготовка модели и данных для сравнения ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(exp_config['model']).to(device)
    model = load_checkpoint(exp_config['model']['checkpoint_path'], model)['model']

    dataset_name = exp_config['dataset']['name']
    stats = DATASET_STATS[dataset_name]
    model_wrapper = ModelWrapper(model, mean=stats['mean'], std=stats['std'])

    # Используем тот же набор данных для честного сравнения
    dataloader, _ = get_dataloader(
        dataset_name=dataset_name,
        num_samples=exp_config.get('num_eval_samples')
    )

    # --- 4. Сравнение с базовыми атаками ---
    logger.info("--- 4. Оценка базовых и лучшей найденной атаки ---")
    with open('configs/baseline_attacks.yaml', 'r') as f:
        baseline_configs = yaml.safe_load(f)

    attacks_to_compare = {
        "Best Found": best_found_config,
        "PGD-10": baseline_configs['pgd_10'],
        "FGSM": baseline_configs['fgsm']
    }

    comparison_results = []
    for name, config in attacks_to_compare.items():
        logger.info(f"Оценка атаки: {name}...")
        results, _ = evaluate_config(config, model_wrapper, dataloader, device)
        results['name'] = name
        comparison_results.append(results)

    df = pd.DataFrame(comparison_results)
    plot_comparison_barchart(df, os.path.join(plots_dir, "attacks_comparison.png"))

    # --- 5. Визуализация примеров лучшей атаки ---
    logger.info("--- 5. Генерация визуальных примеров для лучшей атаки ---")
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    _, examples = evaluate_config(
        best_found_config,
        model_wrapper,
        dataloader,
        device,
        num_examples_to_return=5
    )

    if examples:
        plot_attack_examples(examples, class_names, os.path.join(plots_dir, "best_attack_examples.png"))

    logger.info("Анализ завершен.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Анализ и визуализация результатов поиска атак.")
    parser.add_argument(
        '--results',
        type=str,
        required=True,
        help="Путь к JSON-файлу с результатами поиска (search_results_*.json)."
    )
    args = parser.parse_args()
    main(args.results)