"""
Главный скрипт для запуска поиска лучших состязательных атак.

Этот скрипт выполняет следующие шаги:
1. Загружает конфигурацию эксперимента и пространство поиска.
2. Инициализирует обученную модель и загрузчик данных.
3. В цикле генерирует случайные конфигурации атак.
4. Оценивает каждую конфигурацию на тестовой выборке.
5. Отслеживает и сохраняет лучшую найденную конфигурацию.

Пример запуска:
> python scripts/run_search.py --config configs/experiment_config.yaml
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, List
import gc
import optuna

import torch
import torch.nn as nn
import yaml

from joblib import Parallel, delayed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from advgen.models import get_model
from advgen.training.utils import load_checkpoint
from advgen.core.model_wrapper import ModelWrapper
from advgen.search.samplers import RandomSampler, OptunaSampler
from advgen.search.evaluator import evaluate_config
from advgen.utils.data_loader import get_dataloader, DATASET_STATS
from advgen.utils.logging_setup import setup_logging
from advgen.search.evolution import EvolutionarySampler
from advgen.utils.visualization import TensorBoardLogger


def find_pareto_front(all_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Находит фронт Парето из списка результатов.
    Критерии: (maximize ASR, minimize L2).
    """
    pareto_front = []
    for candidate in all_results:
        if candidate.get('error'):
            continue

        candidate_asr = candidate.get("attack_success_rate", -1.0)
        candidate_l2 = candidate.get("avg_l2_norm", float('inf'))

        is_dominated = False
        for member in pareto_front:
            member_asr = member.get("attack_success_rate", -1.0)
            member_l2 = member.get("avg_l2_norm", float('inf'))

            if (member_asr >= candidate_asr and member_l2 <= candidate_l2) and \
                    (member_asr > candidate_asr or member_l2 < candidate_l2):
                is_dominated = True
                break

        if is_dominated:
            continue

        pareto_front = [
            member for member in pareto_front
            if not ((candidate_asr >= member.get("attack_success_rate", -1.0) and
                     candidate_l2 <= member.get("avg_l2_norm", float('inf'))) and
                    (candidate_asr > member.get("attack_success_rate", -1.0) or
                     candidate_l2 < member.get("avg_l2_norm", float('inf'))))
        ]

        pareto_front.append(candidate)

    return pareto_front


def find_best_trial(trials: List[optuna.Trial]) -> optuna.Trial:
    """
    Находит лучший trial по двум критериям:
    1. Максимизация attack_success_rate.
    2. При равенстве ASR, минимизация avg_l2_norm.
    """
    best_trial = None
    best_asr = -1.0
    best_l2 = float('inf')

    for trial in trials:
        # Пропускаем неудачные (crashed) прогоны
        if "full_results" not in trial.user_attrs:
            continue

        results = trial.user_attrs["full_results"]
        current_asr = results.get("attack_success_rate", -1.0)
        current_l2 = results.get("avg_l2_norm", float('inf'))

        if current_asr > best_asr:
            best_asr = current_asr
            best_l2 = current_l2
            best_trial = trial
        elif current_asr == best_asr and current_l2 < best_l2:
            best_l2 = current_l2
            best_trial = trial

    return best_trial


def main(config_path: str):
    """Основная функция для запуска поиска."""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info(f"Загрузка конфигурации эксперимента из: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        exp_config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Используемое устройство: {device}")

    logger.info(f"Создание модели: {exp_config['model']['name']}")
    model = get_model(exp_config['model']).to(device)

    logger.info(f"Загрузка весов из: {exp_config['model']['checkpoint_path']}")
    checkpoint_data = load_checkpoint(exp_config['model']['checkpoint_path'], model)
    model = checkpoint_data['model']

    dataset_name = exp_config['dataset']['name']
    stats = DATASET_STATS[dataset_name]
    model_wrapper = ModelWrapper(model, mean=stats['mean'], std=stats['std'])

    logger.info("Подготовка загрузчика данных для оценки...")

    search_config = exp_config.get('search', {})
    sampler_type = search_config.get('sampler', 'random')
    num_trials = exp_config['num_trials']

    num_workers = exp_config['dataset'].get('num_workers', 4)
    if sampler_type == 'evolutionary':
        logger.warning(
            "Отключение воркеров DataLoader (num_workers=0) для эволюционного поиска во избежание конфликтов мультипроцессинга.")
        num_workers = 0

    dataloader, _ = get_dataloader(
        dataset_name=dataset_name,
        data_dir=exp_config['dataset']['data_dir'],
        batch_size=exp_config['dataset']['batch_size'],
        num_samples=exp_config.get('num_eval_samples'),
        shuffle=False,
        num_workers=num_workers
    )

    all_trials_results = []
    best_config_info = {"attack_success_rate": -1.0}

    if sampler_type == 'optuna':
        logger.info("Запуск поиска с использованием Optuna (байесовская оптимизация).")
        optuna_sampler = OptunaSampler(exp_config['search_space_path'])

        def objective(trial: optuna.Trial):
            logger.info(f"--- Прогон {trial.number + 1}/{num_trials} (Optuna) ---")

            attack_config = optuna_sampler.sample(trial, norm=exp_config['norm'])
            attack_config['epsilon'] = exp_config['epsilon']
            attack_config['steps'] = exp_config['steps']

            results, _ = evaluate_config(attack_config, model_wrapper, dataloader, device)
            trial.set_user_attr("full_results", results)

            asr = results.get("attack_success_rate", -1.0)
            l2_norm = results.get("avg_l2_norm", float('inf'))
            return asr, l2_norm

        study = optuna.create_study(directions=["maximize", "minimize"])
        study.optimize(objective, n_trials=num_trials)

        pareto_front_trials = study.best_trials
        logger.info(f"Найдено {len(pareto_front_trials)} оптимальных по Парето решений.")

        pareto_front_results = [t.user_attrs.get("full_results") for t in pareto_front_trials if
                                "full_results" in t.user_attrs]
        all_trials_results = [t.user_attrs.get("full_results") for t in study.trials if "full_results" in t.user_attrs]

        best_trial = find_best_trial(pareto_front_trials)
        if best_trial:
            best_config_info = best_trial.user_attrs.get("full_results", {})

    elif sampler_type == 'random':
        logger.info("Запуск поиска с использованием Random Sampler (случайный перебор).")
        sampler = RandomSampler(exp_config['search_space_path'])

        for i in range(num_trials):
            logger.info(f"--- Прогон {i + 1}/{num_trials} (Random) ---")
            attack_config = sampler.sample(norm=exp_config['norm'])
            attack_config['epsilon'] = exp_config['epsilon']
            attack_config['steps'] = exp_config['steps']

            results = evaluate_config(attack_config, model_wrapper, dataloader, device)
            all_trials_results.append(results)

            current_asr = results.get("attack_success_rate", -1.0)

            best_asr = best_config_info.get("attack_success_rate", -1.0)
            best_l2 = best_config_info.get("avg_l2_norm", float('inf'))
            current_l2 = results.get("avg_l2_norm", float('inf'))

            is_new_best = False
            update_reason = ""
            if current_asr > best_asr:
                is_new_best = True
                update_reason = f"ASR улучшен ({best_asr:.4f} -> {current_asr:.4f})"
            elif current_asr == best_asr and current_l2 < best_l2:
                is_new_best = True
                update_reason = f"ASR тот же, но L2 норма меньше ({best_l2:.4f} -> {current_l2:.4f})"

            if is_new_best:
                best_config_info = results
                logger.info(
                    f"*** Найдена новая лучшая конфигурация! "
                    f"Причина: {update_reason} ***"
                )

        logger.info("Поиск фронта Парето из всех случайных прогонов...")
        pareto_front_results = find_pareto_front(all_trials_results)
        logger.info(f"Найдено {len(pareto_front_results)} оптимальных по Парето решений.")

        if pareto_front_results:
            best_config_info = max(pareto_front_results, key=lambda r: r.get('attack_success_rate', -1.0))
        else:
            best_config_info = {"attack_success_rate": -1.0}
    elif sampler_type == 'evolutionary':
        logger.info("Запуск поиска с использованием Evolutionary Sampler (генетический алгоритм).")

        # Получаем параметры из конфига
        evo_config = search_config.get('params', {})
        population_size = evo_config.get('population_size', 50)
        num_generations = exp_config.get('num_trials')  # Переиспользуем num_trials как num_generations

        # Создаем директорию для логов TensorBoard
        results_dir = exp_config.get('results_dir', './results')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(results_dir, f"evolution_logs_{timestamp}")
        os.makedirs(log_dir, exist_ok=True)
        logger.info(f"Логи TensorBoard будут сохраняться в: {log_dir}")

        # Инициализируем сэмплер и логгер
        sampler = EvolutionarySampler(
            search_space_path=exp_config['search_space_path'],
            population_size=population_size,
            mutation_rate=evo_config.get('mutation_rate', 0.1),
            tournament_size=evo_config.get('tournament_size', 3),
            norm=exp_config['norm']
        )
        tb_logger = TensorBoardLogger(log_dir)

        sampler.initialize_population()

        best_overall_fitness = -1.0
        best_overall_config = {}

        # Основной цикл по поколениям
        for gen in range(num_generations):
            logger.info(f"--- Поколение {gen + 1}/{num_generations} ---")

            # Дополняем конфиги общими параметрами
            current_population_configs = []
            for individual in sampler.population:
                config = individual.copy()
                config['epsilon'] = exp_config['epsilon']
                config['steps'] = exp_config['steps']
                current_population_configs.append(config)

            # --- Параллельная оценка всей популяции ---
            logger.info(f"Оценка {len(current_population_configs)} индивидов...")

            tasks_with_progress = []
            for i, conf in enumerate(current_population_configs):
                progress_str = f"Индивид {i + 1}/{len(current_population_configs)}"
                task = delayed(evaluate_config)(
                    conf,
                    model_wrapper,
                    dataloader,
                    device,
                    progress_info=progress_str
                )
                tasks_with_progress.append(task)

            results_list = Parallel(n_jobs=1)(tasks_with_progress)

            fitness_scores = [res[0].get('attack_success_rate', -1.0) for res in results_list]
            all_trials_results.extend([res[0] for res in results_list])

            # Логирование в TensorBoard
            tb_logger.log_generation_stats(sampler.population, fitness_scores, gen)

            # Находим и сохраняем лучшего в этом поколении
            best_idx = fitness_scores.index(max(fitness_scores))
            best_fitness_in_gen = fitness_scores[best_idx]
            best_config_in_gen = sampler.population[best_idx]

            tb_logger.log_best_individual(best_config_in_gen, best_fitness_in_gen, gen)

            # Обновляем лучшего за все время
            if best_fitness_in_gen > best_overall_fitness:
                best_overall_fitness = best_fitness_in_gen
                best_overall_config = results_list[best_idx][0]
                logger.info(f"*** Найдена новая лучшая конфигурация! ASR: {best_overall_fitness:.4f} ***")

            # Эволюция
            sampler.evolve(fitness_scores)

        best_config_info = best_overall_config
        tb_logger.close()
    else:
        raise ValueError(f"Неизвестный тип сэмплера: '{sampler_type}'. Доступны: 'random', 'optuna'.")

    results_dir = exp_config.get('results_dir', './results')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"search_results_{timestamp}.json"
    results_filepath = os.path.join(results_dir, results_filename)
    logger.info(f"Результаты будут сохраняться в: {results_filepath}")

    final_results_summary = {
        "best_performing_attack": best_config_info,
        "pareto_front": pareto_front_results,
        "all_trials": all_trials_results,
        "experiment_config": exp_config
    }

    with open(results_filepath, 'w', encoding='utf-8') as f:
        json.dump(final_results_summary, f, indent=4, ensure_ascii=False)

    logger.info("--- Поиск завершен ---")
    if best_config_info["attack_success_rate"] > -1.0:
        logger.info(f"Лучший Attack Success Rate (ASR): {best_config_info['attack_success_rate']:.4f}")
        logger.info("Лучшая найденная конфигурация:")

        best_config_str = yaml.dump(best_config_info.get("processed_config", {}), indent=2)
        for line in best_config_str.splitlines():
            logger.info(f"  {line}")
    else:
        logger.warning("Не удалось найти ни одной успешной конфигурации.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Скрипт для поиска состязательных атак.")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help="Путь к YAML-файлу с конфигурацией эксперимента."
    )
    args = parser.parse_args()
    main(args.config)