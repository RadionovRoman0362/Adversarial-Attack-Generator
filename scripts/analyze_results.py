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
import pickle

import optuna
from optuna.visualization import plot_param_importances, plot_slice, plot_parallel_coordinate
import os
import sys
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from advgen.core.attack_runner import AttackRunner
from advgen.core.model_wrapper import ModelWrapper
from advgen.models import get_model
from advgen.search.evaluator import evaluate_config, _preprocess_config
from advgen.training.utils import load_checkpoint
from advgen.utils.data_loader import get_dataloader, DATASET_STATS
from advgen.utils.logging_setup import setup_logging


# --- Функции для визуализации ---

def plot_component_dynamics(df: pd.DataFrame, save_path: str):
    """Строит график динамики популярности компонентов градиента."""
    if 'generation' not in df.columns or 'processed_config' not in df.columns:
        return

    def get_grad_components_from_config(config):
        chain = []
        curr = config.get('gradient', {})
        while curr:
            chain.append(curr.get('name'))
            curr = curr.get('wrapped')
        return chain

    dynamics_data = []
    for gen, group in df.groupby('generation'):
        all_components = []
        for config in group['processed_config']:
            all_components.extend(get_grad_components_from_config(config))

        counts = pd.Series(all_components).value_counts()
        counts.name = gen
        dynamics_data.append(counts)

    dynamics_df = pd.DataFrame(dynamics_data).fillna(0).sort_index()
    dynamics_df = dynamics_df.reindex(sorted(dynamics_df.columns), axis=1)  # Сортируем для красивой легенды

    fig, ax = plt.subplots(figsize=(14, 8))
    dynamics_df.plot(kind='area', stacked=True, ax=ax, colormap='viridis')

    ax.set_title("Динамика популярности компонентов градиента по поколениям", fontsize=16)
    ax.set_xlabel("Поколение")
    ax.set_ylabel("Количество в популяции")
    ax.legend(title='Компоненты', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info(f"График динамики компонентов сохранен в: {save_path}")

def plot_evolution_history(df: pd.DataFrame, save_path: str):
    """Строит график истории улучшения для эволюционного алгоритма."""
    if 'generation' not in df.columns:
        logging.warning("Нет данных о поколениях для построения графика истории эволюции.")
        return

    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")

    best_per_gen = df.groupby('generation')['attack_success_rate'].max()
    best_so_far = best_per_gen.cummax()

    plt.plot(best_so_far.index, best_so_far.values, marker='o', linestyle='-', label='Лучший ASR на данный момент')

    sns.scatterplot(data=df, x='generation', y='attack_success_rate', color='gray', alpha=0.2, label='ASR индивидов')

    plt.title("История оптимизации (Эволюционный поиск)", fontsize=16)
    plt.xlabel("Поколение")
    plt.ylabel("Attack Success Rate (ASR)")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    logging.info(f"График истории эволюции сохранен в: {save_path}")

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


def plot_optimization_history(df: pd.DataFrame, save_path: str):
    """
    Строит график истории оптимизации: как менялся лучший результат со временем.
    """
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")

    # Находим лучший результат на каждом шаге (прогоне)
    df['best_so_far'] = df['attack_success_rate'].cummax()

    plt.plot(df.index, df['best_so_far'], marker='o', linestyle='-', label='Лучший ASR на данный момент')
    plt.scatter(df.index, df['attack_success_rate'], color='gray', alpha=0.5, label='ASR текущего прогона')

    plt.title("История оптимизации поиска", fontsize=16)
    plt.xlabel("Номер прогона (Trial)")
    plt.ylabel("Attack Success Rate (ASR)")
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path)
    plt.close()
    logging.info(f"График истории оптимизации сохранен в: {save_path}")


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
    """
    Строит и сохраняет информативную диаграмму для сравнения атак.
    Использует два подграфика для метрик с разным масштабом.
    """
    sns.set_theme(style="whitegrid")

    # Создаем фигуру с двумя подграфиками (2 строки, 1 колонка)
    # sharex=True связывает их по оси X, что очень удобно для сравнения
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    fig.suptitle("Сравнение эффективности и заметности атак", fontsize=18, y=0.95)

    # --- График 1: Attack Success Rate (Главная метрика) ---
    ax1 = axes[0]
    sns.barplot(
        data=df,
        x='name',
        y='attack_success_rate',
        ax=ax1,
        hue='name',
        palette='viridis',
        legend=False
    )
    ax1.set_title("Эффективность (чем выше, тем лучше)", fontsize=14)
    ax1.set_ylabel("Attack Success Rate (ASR)")
    ax1.set_xlabel("") # Убираем подпись X у верхнего графика
    ax1.set_ylim(0, 1.05) # Ось Y от 0 до 100%

    # Добавляем текстовые метки на столбцы для наглядности
    for p in ax1.patches:
        ax1.annotate(f"{p.get_height():.3f}",
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center',
                     xytext=(0, 9),
                     textcoords='offset points')

    # --- График 2: Нормы возмущений (Заметность) ---
    ax2 = axes[1]

    # Готовим данные для сдвоенной диаграммы (только нормы)
    df_norms = df.melt(
        id_vars='name',
        value_vars=['avg_linf_norm', 'avg_l2_norm'],
        var_name='Norm Type',
        value_name='Average Perturbation'
    )

    sns.barplot(
        data=df_norms,
        x='name',
        y='Average Perturbation',
        hue='Norm Type',
        ax=ax2,
        palette='plasma'
    )
    ax2.set_title("Заметность (чем ниже, тем лучше)", fontsize=14)
    ax2.set_ylabel("Средняя норма возмущения")
    ax2.set_xlabel("Название атаки")

    # Вращаем подписи на оси X, если они длинные
    plt.xticks(rotation=15, ha='right')

    # Улучшаем расположение
    plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # Оставляем место для общего заголовка

    plt.savefig(save_path)
    plt.close()
    logging.info(f"График сравнения сохранен в: {save_path}")


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

    all_trials_df = pd.DataFrame(results_data['all_trials']).dropna(subset=['attack_success_rate'])
    if exp_config.get('search', {}).get('sampler') == 'optuna':
        logger.info("--- 2. Генерация графика истории оптимизации ---")
        if not all_trials_df.empty:
            plot_optimization_history(all_trials_df, os.path.join(plots_dir, "optuna_history.png"))
        else:
            logger.warning("Нет данных для построения графика истории оптимизации.")

        logger.info("--- 2.5. Анализ важности гиперпараметров Optuna ---")
        optuna_filename = base_filename.replace("search_results_", "")
        study_path = os.path.join(analysis_dir, f"optuna_study_{optuna_filename}.pkl")
        if os.path.exists(study_path):
            with open(study_path, "rb") as f_in:
                study = pickle.load(f_in)

            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if not completed_trials:
                logger.warning("В исследовании Optuna нет завершенных прогонов. Пропускаем анализ параметров.")
            else:
                existing_params = set(completed_trials[0].params.keys())

                try:
                    fig = plot_param_importances(study, target=lambda t: t.values[0], target_name="ASR")
                    fig.update_layout(title="Важность гиперпараметров для ASR (Optuna)")
                    fig.write_image(os.path.join(plots_dir, "optuna_param_importances_asr.png"))

                    fig = plot_param_importances(study, target=lambda t: t.values[1], target_name="L2 Norm")
                    fig.update_layout(title="Важность гиперпараметров для L2-нормы (Optuna)")
                    fig.write_image(os.path.join(plots_dir, "optuna_param_importances_l2.png"))

                    logger.info("Графики важности параметров Optuna сохранены.")
                except Exception as e:
                    logger.warning(f"Не удалось построить график важности параметров: {e}")

                logger.info("--- 2.6. Анализ срезов для ключевых компонентов (Optuna) ---")
                try:
                    key_component_params = [
                        'loss_',
                        'gradient_terminal',
                        'gradient_slot_1',
                        'gradient_slot_2',
                        'gradient_slot_3',
                        'scheduler'
                        'initializer',
                        'projector',
                        'updater'
                    ]

                    params_to_plot_slice = [p for p in key_component_params if p in existing_params]

                    for param in params_to_plot_slice:
                        fig_asr = plot_slice(study, params=[param], target=lambda t: t.values[0], target_name="ASR")
                        fig_asr.update_layout(title=f"Влияние компонента '{param}' на ASR")
                        fig_asr.write_image(os.path.join(plots_dir, f"optuna_slice_{param}_asr.png"))

                        fig_l2 = plot_slice(study, params=[param], target=lambda t: t.values[1], target_name="L2 Norm")
                        fig_l2.update_layout(title=f"Влияние компонента '{param}' на L2-норму")
                        fig_l2.write_image(os.path.join(plots_dir, f"optuna_slice_{param}_l2.png"))

                    logger.info("Графики срезов для компонентов сохранены.")

                except Exception as e:
                    logger.warning(f"Не удалось построить графики срезов: {e}")

                logger.info("--- 2.7. Анализ параллельных координат (Optuna) ---")
                try:
                    key_component_params = [
                        'loss_',
                        'gradient_terminal',
                        'gradient_slot_1',
                        'gradient_slot_2',
                        'gradient_slot_3',
                        'scheduler'
                        'initializer',
                        'projector',
                        'updater'
                    ]

                    params_to_plot_parallel = [p for p in key_component_params if p in existing_params]

                    if params_to_plot_parallel:
                        fig_asr = plot_parallel_coordinate(study, params=params_to_plot_parallel, target=lambda t: t.values[0],
                                                           target_name="ASR")
                        fig_asr.update_layout(title="Параллельные координаты для поиска лучшего ASR")
                        fig_asr.write_image(os.path.join(plots_dir, "optuna_parallel_coord_asr.png"))

                        fig_l2 = plot_parallel_coordinate(study, params=params_to_plot_parallel, target=lambda t: t.values[1],
                                                          target_name="L2 Norm")
                        fig_l2.update_layout(title="Параллельные координаты для поиска меньшей L2-нормы")
                        fig_l2.write_image(os.path.join(plots_dir, "optuna_parallel_coord_l2.png"))

                        logger.info("Графики параллельных координат сохранены.")

                except Exception as e:
                    logger.warning(f"Не удалось построить график параллельных координат: {e}")
        else:
            logger.warning(f"Файл Optuna Study не найден по пути: {study_path}")
    elif exp_config.get('search', {}).get('sampler') == 'evolutionary':
        logger.info("--- 2. Генерация графика истории эволюции ---")
        if not all_trials_df.empty:
            plot_evolution_history(all_trials_df, os.path.join(plots_dir, "evolution_history.png"))
            plot_component_dynamics(all_trials_df, os.path.join(plots_dir, "evolution_dynamics.png"))
        else:
            logger.warning("Нет данных для построения графика истории оптимизации и динамики компонентов.")

    if "pareto_front" in results_data and "all_trials" in results_data:
        logger.info("--- 2.5. Визуализация фронта Парето ---")
        all_trials_df = pd.DataFrame(results_data['all_trials']).dropna()
        pareto_front_df = pd.DataFrame(results_data['pareto_front']).dropna()

        if not pareto_front_df.empty:
            plot_pareto_front(all_trials_df, pareto_front_df, os.path.join(plots_dir, "pareto_front.png"))

    logger.info("--- 3. Подготовка моделей и данных для сравнения ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_wrappers = []
    model_configs = []
    if 'models' in exp_config:
        model_configs.extend(exp_config['models'])
    elif 'model' in exp_config:
        model_configs.append(exp_config['model'])
    else:
        raise ValueError("...")

    dataset_name = exp_config['dataset']['name']
    stats = DATASET_STATS[dataset_name]

    for model_config in model_configs:
        model = get_model(model_config).to(device)
        model = load_checkpoint(model_config['checkpoint_path'], model)['model']
        model_wrapper = ModelWrapper(model, mean=stats['mean'], std=stats['std'])
        model_wrapper.eval()
        model_wrappers.append(model_wrapper)

    surrogate_model_wrapper = model_wrappers[0]

    full_eval_dataloader, _ = get_dataloader(
        dataset_name=dataset_name,
        num_samples=exp_config.get('num_eval_samples')
    )

    visual_dataloader, _ = get_dataloader(
        dataset_name=dataset_name,
        num_samples=100,
        batch_size=10,
        shuffle=False
    )

    with open('configs/baseline_attacks.yaml', 'r') as f:
        baseline_configs = yaml.safe_load(f)

    attacks_to_compare = {
        "Best Found": best_found_config,
        "PGD-10": baseline_configs['pgd_10'],
        "FGSM": baseline_configs['fgsm']
    }

    logger.info("--- 4. Оценка эффективности атак ---")
    comparison_results = []
    for name, config in attacks_to_compare.items():
        logger.info(f"Оценка атаки: {name}...")
        results, _ = evaluate_config(config, model_wrappers, full_eval_dataloader, device)
        results['name'] = name
        comparison_results.append(results)

    df = pd.DataFrame(comparison_results)
    plot_comparison_barchart(df, os.path.join(plots_dir, "attacks_comparison.png"))

    logger.info("--- 5. Генерация визуальных примеров ---")

    num_examples_to_plot = 5
    correctly_classified_examples = []

    logger.info(f"Поиск {num_examples_to_plot} правильно классифицированных примеров для визуализации...")
    for images, labels in visual_dataloader:
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            clean_logits = surrogate_model_wrapper(images)
            clean_preds = torch.argmax(clean_logits, dim=1)

        correct_mask = (clean_preds == labels)

        for i in range(len(images)):
            if correct_mask[i]:
                correctly_classified_examples.append({
                    "original_image": images[i].cpu(),
                    "label": labels[i].cpu(),
                    "clean_pred": clean_preds[i].cpu()
                })

        if len(correctly_classified_examples) >= num_examples_to_plot:
            break

    visual_examples_base = correctly_classified_examples[:num_examples_to_plot]

    if not visual_examples_base:
        logger.warning("Не удалось найти ни одного правильно классифицированного примера для визуализации.")
    else:
        logger.info(f"Найдено {len(visual_examples_base)} примеров. Генерация атак для них...")
        original_images_batch = torch.stack([ex["original_image"] for ex in visual_examples_base]).to(device)
        labels_batch = torch.stack([ex["label"] for ex in visual_examples_base]).to(device)

        for name, config in attacks_to_compare.items():
            logger.info(f"Генерация примеров для атаки: {name}...")
            config = _preprocess_config(config)
            runner = AttackRunner(config, model_wrappers)
            adv_images_batch = runner.attack(surrogate_model_wrapper, original_images_batch, labels_batch)

            with torch.no_grad():
                adv_logits = surrogate_model_wrapper(adv_images_batch)
                adv_preds = torch.argmax(adv_logits, dim=1)

            for i in range(len(visual_examples_base)):
                visual_examples_base[i][f"adv_image_{name}"] = adv_images_batch[i].cpu()
                visual_examples_base[i][f"adv_pred_{name}"] = adv_preds[i].cpu().item()

        if exp_config['dataset']['name'] == 'imagenette':
            class_names = ['tench', 'English springer', 'cassette player', 'chain saw', 'church', 'French horn',
                           'garbage truck', 'gas pump', 'golf ball', 'parachute']
        else:
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        plot_attack_examples(
            visual_examples_base,
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