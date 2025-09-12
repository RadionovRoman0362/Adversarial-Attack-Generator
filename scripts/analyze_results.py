"""
Скрипт для комплексного анализа и визуализации результатов поиска атак.

Функционал:
1. Загружает JSON-файл с результатами, полученный от `run_search.py`.
2. (Для Optuna) Строит графики процесса оптимизации и важности параметров.
3. Оценивает лучшую найденную атаку и базовые атаки (PGD, FGSM) на одном
   и том же наборе данных для честного сравнения.
4. Строит столбчатую диаграмму, сравнивающую атаки по ASR и нормам возмущений.
5. Генерирует и сохраняет визуальные примеры для лучшей атаки И базовых атак,
   позволяя визуально сравнить "заметность" возмущений.

Пример запуска:
> python scripts/analyze_results.py --results results/search_results_optuna_....json
"""

import argparse
import json
import logging
import optuna
import os
import sys
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.models as models
import yaml

# Добавляем корневую директорию проекта в PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from advgen.core.attack_runner import AttackRunner
from advgen.core.model_wrapper import ModelWrapper
from advgen.models import resnet_cifar
from advgen.search.evaluator import evaluate_config, _preprocess_config
from advgen.training.utils import load_checkpoint
from advgen.utils.data_loader import get_dataloader, DATASET_STATS
from advgen.utils.logging_setup import setup_logging


# --- Функции для визуализации ---

def plot_pareto_front(all_trials_df: pd.DataFrame, pareto_front_df: pd.DataFrame, save_path: str):
    """Строит диаграмму рассеяния, выделяя фронт Парето."""
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")

    # Рисуем все точки
    sns.scatterplot(
        data=all_trials_df,
        x='avg_l2_norm',
        y='attack_success_rate',
        color='gray',
        alpha=0.5,
        label='Все прогоны'
    )

    # Выделяем точки с фронта Парето
    sns.scatterplot(
        data=pareto_front_df,
        x='avg_l2_norm',
        y='attack_success_rate',
        color='red',
        s=100,  # Размер точек
        edgecolor='black',
        label='Фронт Парето'
    )

    plt.title("Фронт Парето: компромисс между ASR и L2-нормой", fontsize=16)
    plt.xlabel("Средняя L2-норма (меньше = лучше)")
    plt.ylabel("Attack Success Rate (больше = лучше)")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    logging.info(f"График фронта Парето сохранен в: {save_path}")


def plot_attack_examples(
    visual_examples: List[Dict[str, Any]],
    attack_names: List[str],
    class_names: List[str],
    save_path: str,
):
    """
    Визуализирует и сохраняет примеры атак для нескольких методов.
    Для каждого исходного изображения строится строка:
    [Оригинал, Возмущение(Лучшая), Атака(Лучшая), Возмущение(PGD), Атака(PGD), ...]
    """
    num_examples = len(visual_examples)
    num_attacks = len(attack_names)

    # +1 для оригинала, +1 для каждого возмущения
    num_cols = 1 + 2 * num_attacks

    if num_examples == 0:
        return

    fig, axes = plt.subplots(num_examples, num_cols, figsize=(4 * num_cols, 4 * num_examples))
    fig.suptitle("Визуальное сравнение состязательных атак", fontsize=20, y=1.0)

    for i in range(num_examples):
        example_data = visual_examples[i]
        orig_img_tensor = example_data["original_image"]
        orig_img_np = orig_img_tensor.permute(1, 2, 0).numpy()

        label = class_names[example_data["label"]]
        clean_pred = class_names[example_data["clean_pred"]]

        # 1. Отображаем оригинал
        ax = axes[i, 0] if num_examples > 1 else axes[0]
        ax.imshow(orig_img_np)
        ax.set_title(f"Оригинал #{i+1}\nИстина: {label}\nПредсказание: {clean_pred}", fontsize=12)
        ax.axis('off')

        # 2. Отображаем атаки и возмущения
        for j, attack_name in enumerate(attack_names):
            adv_img_tensor = example_data[f"adv_image_{attack_name}"]
            adv_img_np = adv_img_tensor.permute(1, 2, 0).numpy()
            adv_pred = class_names[example_data[f"adv_pred_{attack_name}"]]

            diff = np.abs(adv_img_np - orig_img_np)
            diff_normalized = (diff - diff.min()) / (diff.max() - diff.min() + 1e-9)

            # Возмущение
            ax_diff = axes[i, 2*j + 1] if num_examples > 1 else axes[2*j + 1]
            ax_diff.imshow(diff_normalized)
            ax_diff.set_title(f"Возмущение\n({attack_name})", fontsize=12)
            ax_diff.axis('off')

            # Атакованное изображение
            ax_adv = axes[i, 2*j + 2] if num_examples > 1 else axes[2*j + 2]
            ax_adv.imshow(adv_img_np)
            ax_adv.set_title(f"Атака ({attack_name})\nПредсказание: {adv_pred}", fontsize=12)
            ax_adv.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Визуальные примеры сохранены в: {save_path}")


def plot_comparison_barchart(df: pd.DataFrame, save_path: str):
    """Строит и сохраняет столбчатую диаграмму для сравнения атак."""
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")

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
    model_name = config['name']
    num_classes = config.get('num_classes', 10)
    pretrained = config.get('pretrained', False)
    if model_name == 'resnet18_cifar':
        return resnet_cifar.resnet18_cifar(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'resnet18_imagenet':
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        return model
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

    analysis_dir = os.path.dirname(results_path)
    base_filename = os.path.splitext(os.path.basename(results_path))[0]
    plots_dir = os.path.join(analysis_dir, f"analysis_{base_filename}")
    os.makedirs(plots_dir, exist_ok=True)
    logger.info(f"Графики и анализ будут сохранены в: {plots_dir}")

    # --- 2. Визуализация процесса поиска (только для Optuna) ---
    if exp_config.get('search', {}).get('sampler') == 'optuna':
        logger.info("--- 2. Генерация графиков анализа Optuna ---")
        try:
            # Optuna не имеет встроенной функции для загрузки study из JSON.
            # Мы "воссоздаем" его, добавляя каждый trial из нашего лог-файла.
            # Это позволяет использовать мощные встроенные функции визуализации Optuna.

            search_directions = exp_config['search'].get('directions', ["maximize"])
            study = optuna.create_study(directions=search_directions)

            all_trials_data = results_data.get('all_trials', [])

            # Перед добавлением нужно убедиться, что все параметры плоские,
            # так как Optuna не работает с вложенными словарями параметров.
            flat_trials_data = []
            for trial_data in all_trials_data:
                if trial_data.get('error'):
                    continue  # Пропускаем неудачные прогоны

                # Используем pandas для "уплощения" вложенного словаря
                flat_params = pd.json_normalize(
                    trial_data['processed_config'], sep='_'
                ).to_dict(orient='records')[0]

                # Определяем, какие значения вернуть (одно или несколько)
                if len(search_directions) > 1:
                    values = [
                        trial_data.get("attack_success_rate", -1.0),
                        trial_data.get("avg_l2_norm", float('inf'))
                    ]
                else:
                    values = [trial_data.get("attack_success_rate", -1.0)]

                flat_trials_data.append({
                    "values": values,
                    "params": flat_params,
                    "user_attrs": {"full_results": trial_data}  # Сохраняем оригинальные данные
                })

            # Добавляем воссозданные trials в study
            for trial_info in flat_trials_data:
                study.add_trial(
                    optuna.create_trial(
                        values=trial_info["values"],
                        params=trial_info["params"],
                        user_attrs=trial_info["user_attrs"]
                    )
                )

            logger.info(f"Study успешно воссоздан с {len(study.trials)} прогонами.")

            # Генерация и сохранение графиков
            # 1. История оптимизации (показывает, как улучшалась метрика со временем)
            fig_history = study.plot_optimization_history()
            fig_history.write_image(os.path.join(plots_dir, "optuna_history.png"))

            # 2. Важность гиперпараметров (какие параметры больше всего влияли на результат)
            # Эта визуализация работает только для однокритериальной оптимизации
            if len(search_directions) == 1:
                try:
                    fig_importance = study.plot_param_importances()
                    fig_importance.write_image(os.path.join(plots_dir, "optuna_importances.png"))
                except Exception as e:
                    logger.warning(f"Не удалось построить график важности параметров: {e}")

            # 3. Срезы параметров (показывает, как конкретный параметр влияет на результат)
            # Выберем несколько интересных параметров для визуализации
            slice_params_to_plot = [
                'loss_name',
                'gradient_name',
                'scheduler_cosine_params_initial_step_size',
                'updater_name'
            ]
            for param in slice_params_to_plot:
                # Проверяем, существует ли параметр в исследовании, прежде чем строить график
                if param in study.best_params or any(param in t.params for t in study.trials):
                    try:
                        fig_slice = study.plot_slice(params=[param])
                        fig_slice.write_image(os.path.join(plots_dir, f"optuna_slice_{param}.png"))
                    except Exception as e:
                        logger.warning(f"Не удалось построить срез для параметра '{param}': {e}")
                else:
                    logger.debug(f"Параметр '{param}' не найден в исследовании, пропускаем срез.")

            logger.info("Графики анализа Optuna успешно сохранены.")

        except Exception as e:
            logger.error(f"Произошла ошибка при генерации графиков Optuna: {e}")

    if "pareto_front" in results_data and "all_trials" in results_data:
        logger.info("--- 2.5. Визуализация фронта Парето ---")
        all_trials_df = pd.DataFrame(results_data['all_trials']).dropna()
        pareto_front_df = pd.DataFrame(results_data['pareto_front']).dropna()

        if not pareto_front_df.empty:
            plot_pareto_front(all_trials_df, pareto_front_df, os.path.join(plots_dir, "pareto_front.png"))

    # --- 3. Подготовка к сравнению атак ---
    logger.info("--- 3. Подготовка модели и данных для сравнения ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(exp_config['model']).to(device)
    model = load_checkpoint(exp_config['model']['checkpoint_path'], model)['model']

    dataset_name = exp_config['dataset']['name']
    stats = DATASET_STATS[dataset_name]
    model_wrapper = ModelWrapper(model, mean=stats['mean'], std=stats['std'])

    full_eval_dataloader, _ = get_dataloader(
        dataset_name=dataset_name,
        num_samples=exp_config.get('num_eval_samples')
    )

    # Отдельный, маленький DataLoader только для генерации картинок
    visual_dataloader, _ = get_dataloader(
        dataset_name=dataset_name,
        num_samples=10, # Берем 10 картинок, чтобы было из чего выбрать
        batch_size=5,   # Маленький батч
        shuffle=False
    )

    with open('configs/baseline_attacks.yaml', 'r') as f:
        baseline_configs = yaml.safe_load(f)

    attacks_to_compare = {
        "Best Found": best_found_config,
        "PGD-10": baseline_configs['pgd_10'],
        "FGSM": baseline_configs['fgsm']
    }

    # --- 4. Оценка эффективности (ASR, нормы) на большом датасете ---
    logger.info("--- 4. Оценка эффективности атак ---")
    comparison_results = []
    for name, config in attacks_to_compare.items():
        logger.info(f"Оценка атаки: {name}...")
        results, _ = evaluate_config(config, model_wrapper, full_eval_dataloader, device)
        results['name'] = name
        comparison_results.append(results)

    df = pd.DataFrame(comparison_results)
    plot_comparison_barchart(df, os.path.join(plots_dir, "attacks_comparison.png"))

    # --- 5. Генерация примеров для визуального сравнения ---
    logger.info("--- 5. Генерация визуальных примеров ---")

    # Берем один батч из маленького даталоадера
    visual_images, visual_labels = next(iter(visual_dataloader))
    visual_images, visual_labels = visual_images.to(device), visual_labels.to(device)

    visual_examples = []
    num_examples_to_plot = 5

    # Получаем предсказания на чистых данных
    with torch.no_grad():
        clean_logits = model_wrapper(visual_images)
        clean_preds = torch.argmax(clean_logits, dim=1)

    # Инициализируем структуру для хранения примеров
    for i in range(len(visual_images)):
        visual_examples.append({
            "original_image": visual_images[i].cpu(),
            "label": visual_labels[i].cpu().item(),
            "clean_pred": clean_preds[i].cpu().item(),
        })

    # Запускаем каждую атаку на этом батче и сохраняем результаты
    for name, config in attacks_to_compare.items():
        logger.info(f"Генерация примеров для атаки: {name}...")
        config = _preprocess_config(config) # Не забываем про обработку
        runner = AttackRunner(config)
        adv_images_batch = runner.attack(model_wrapper, visual_images, visual_labels)

        with torch.no_grad():
            adv_logits = model_wrapper(adv_images_batch)
            adv_preds = torch.argmax(adv_logits, dim=1)

        # Добавляем результаты в нашу структуру
        for i in range(len(visual_images)):
            visual_examples[i][f"adv_image_{name}"] = adv_images_batch[i].cpu()
            visual_examples[i][f"adv_pred_{name}"] = adv_preds[i].cpu().item()

    if exp_config['dataset']['name'] == 'imagenette':
        class_names = [
            'tench', 'English springer', 'cassette player', 'chain saw',
            'church', 'French horn', 'garbage truck', 'gas pump',
            'golf ball', 'parachute'
        ]
    else:  # По умолчанию CIFAR-10
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    plot_attack_examples(
        visual_examples[:num_examples_to_plot],
        list(attacks_to_compare.keys()),
        class_names,
        os.path.join(plots_dir, "visual_attack_comparison.png")
    )

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