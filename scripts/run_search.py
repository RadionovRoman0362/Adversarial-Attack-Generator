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
import optuna
import pickle

import torch
import yaml

from joblib import Parallel, delayed
from pathlib import Path

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
from advgen.utils.reproducibility import set_seed


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


def main(config_path: str, resume_path: str = None):
    """Основная функция для запуска поиска."""
    setup_logging()
    logger = logging.getLogger(__name__)

    if resume_path:
        logger.info(f"--- ВОЗОБНОВЛЕНИЕ РАБОТЫ С ЧЕКПОИНТА: {resume_path} ---")
        with open(resume_path, 'rb') as f:
            state = pickle.load(f)

        exp_config = state['exp_config']
        config_path = state['config_path']
        logger.info(f"Конфигурация эксперимента загружена из чекпоинта (оригинал: {config_path})")

        all_trials_results = state['all_trials_results']

        sampler_type = exp_config.get('search', {}).get('sampler', 'random')
        if sampler_type == 'optuna':
            study = state['sampler_state']
            start_trial = len(study.trials)
            best_config_info = find_best_trial(study.best_trials).user_attrs.get(
                "full_results",
                {}
            ) if study.best_trials else {
                "attack_success_rate": -1.0
            }
            logger.info(f"Состояние Optuna восстановлено. Продолжение с прогона #{start_trial + 1}.")
        elif sampler_type == 'evolutionary':
            sampler_state = state['sampler_state']
            start_generation = state['start_generation']
            best_config_info = state['best_config_info']
            logger.info(
                f"Состояние Evolutionary Sampler восстановлено. Продолжение с поколения #{start_generation + 1}.")
        else:  # random
            start_trial = state['start_trial']
            best_config_info = state['best_config_info']
            logger.info(f"Состояние Random Search восстановлено. Продолжение с прогона #{start_trial + 1}.")

    else:
        logger.info(f"--- НАЧАЛО НОВОГО ПОИСКА ---")
        logger.info(f"Загрузка конфигурации эксперимента из: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            exp_config = yaml.safe_load(f)

        all_trials_results = []
        best_config_info = {"attack_success_rate": -1.0}
        start_trial = 0
        start_generation = 0
        study = None

    seed = exp_config.get('seed')
    if seed is not None:
        logger.info(f"Установка глобального random seed: {seed}")
        set_seed(seed)
    else:
        logger.warning("Random seed не указан в конфигурации. Результаты могут быть невоспроизводимы.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Используемое устройство: {device}")

    logger.info("Создание и загрузка моделей для атаки...")
    model_wrappers = []
    dataset_name = exp_config['dataset']['name']
    stats = DATASET_STATS[dataset_name]

    model_configs = []
    if 'models' in exp_config:
        logger.info(f"Обнаружено {len(exp_config['models'])} моделей в конфигурации.")
        model_configs.extend(exp_config['models'])
    elif 'model' in exp_config:
        logger.info("Обнаружена одна модель в конфигурации.")
        model_configs.append(exp_config['model'])
    else:
        raise ValueError("В конфигурации эксперимента не найдены ключи 'model' или 'models'.")

    for model_config in model_configs:
        logger.info(f"  - Загрузка модели из: {model_config['checkpoint_path']}")
        model = get_model(model_config).to(device)
        model = load_checkpoint(model_config['checkpoint_path'], model)['model']
        model_wrapper = ModelWrapper(model, mean=stats['mean'], std=stats['std'])
        model_wrapper.eval()
        model_wrappers.append(model_wrapper)

    logger.info(f"Всего загружено {len(model_wrappers)} моделей для оценки атак.")

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

    results_dir = exp_config.get('results_dir', './results')
    os.makedirs(results_dir, exist_ok=True)

    if resume_path:
        checkpoint_path = Path(resume_path)
        timestamp = checkpoint_path.stem.replace('checkpoint_', '')
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    checkpoint_filepath = os.path.join(results_dir, f"checkpoint_{timestamp}.pkl")
    logger.info(f"Файл чекпоинта будет сохраняться в: {checkpoint_filepath}")

    if sampler_type == 'optuna':
        logger.info("Запуск поиска с использованием Optuna (байесовская оптимизация).")
        optuna_sampler = OptunaSampler(exp_config['search_space_path'])

        def objective(trial: optuna.Trial):
            logger.info(f"--- Прогон {trial.number + 1}/{num_trials} (Optuna) ---")

            attack_config = optuna_sampler.sample(trial, norm=exp_config['norm'])
            attack_config['epsilon'] = exp_config['epsilon']
            attack_config['steps'] = exp_config['steps']

            results, _ = evaluate_config(attack_config, model_wrappers, dataloader, device,
                                         progress_info=f"Прогон {trial.number + 1}/{num_trials}")
            trial.set_user_attr("full_results", results)

            asr = results.get("attack_success_rate", -1.0)
            l2_norm = results.get("avg_l2_norm", float('inf'))

            if isinstance(study.directions, list) and len(study.directions) > 1:
                return asr, l2_norm
            else:
                return asr

        if study is None:
            from optuna.samplers import TPESampler
            seed = exp_config.get('seed')
            sampler_for_study = TPESampler(seed=seed) if seed is not None else TPESampler()

            search_directions = search_config.get('direction', 'maximize')
            if not isinstance(search_directions, list):
                search_directions = [search_directions]  # Приводим к списку для единообразия

            study = optuna.create_study(directions=search_directions, sampler=sampler_for_study)

        remaining_trials = num_trials - start_trial

        if remaining_trials > 0:
            logger.info(f"Запуск {remaining_trials} прогонов Optuna (с {start_trial + 1} по {num_trials}).")
            for i in range(remaining_trials):
                study.optimize(objective, n_trials=1)

                current_trial_num = start_trial + i + 1
                state_to_save = {
                    'exp_config': exp_config,
                    'config_path': config_path,
                    'sampler_state': study,
                    'all_trials_results': [t.user_attrs.get("full_results") for t in study.trials if
                                           "full_results" in t.user_attrs]
                }
                with open(checkpoint_filepath, 'wb') as f:
                    pickle.dump(state_to_save, f)
                logger.debug(f"Чекпоинт для прогона Optuna #{current_trial_num} сохранен.")
        else:
            logger.info("Все прогоны Optuna уже завершены согласно чекпоинту. Пропускаем оптимизацию.")

        study_path = os.path.join(results_dir, f"optuna_study_{timestamp}.pkl")
        with open(study_path, "wb") as f_out:
            pickle.dump(study, f_out)
        logger.info(f"Финальный объект Optuna Study сохранен в: {study_path}")

        all_trials_results = [t.user_attrs.get("full_results") for t in study.trials if "full_results" in t.user_attrs]

        pareto_front_trials = study.best_trials
        logger.info(f"Поиск завершен. Найдено {len(pareto_front_trials)} оптимальных по Парето решений.")

        best_trial = find_best_trial(pareto_front_trials)
        if best_trial:
            best_config_info = best_trial.user_attrs.get("full_results", {})

        pareto_front_results = [t.user_attrs.get("full_results") for t in pareto_front_trials if
                                "full_results" in t.user_attrs]

    elif sampler_type == 'random':
        logger.info("Запуск поиска с использованием Random Sampler (случайный перебор).")
        sampler = RandomSampler(exp_config['search_space_path'])
        remaining_trials = num_trials - start_trial

        if remaining_trials > 0:
            logger.info(f"Запуск {remaining_trials} прогонов Random Search (с {start_trial + 1} по {num_trials}).")

            for i in range(start_trial, num_trials):
                current_trial_num = i + 1
                progress_info = f"Прогон {current_trial_num}/{num_trials}"

                logger.info(f"--- {progress_info} (Random) ---")
                attack_config = sampler.sample(norm=exp_config['norm'])
                attack_config['epsilon'] = exp_config['epsilon']
                attack_config['steps'] = exp_config['steps']

                results, _ = evaluate_config(attack_config, model_wrappers, dataloader, device,
                                             progress_info=progress_info)
                all_trials_results.append(results)

                current_asr = results.get("attack_success_rate", -1.0)
                current_l2 = results.get("avg_l2_norm", float('inf'))
                best_asr = best_config_info.get("attack_success_rate", -1.0)
                best_l2 = best_config_info.get("avg_l2_norm", float('inf'))
                is_new_best = False
                update_reason = ""

                if current_asr > best_asr:
                    is_new_best = True
                    update_reason = f"ASR улучшен ({best_asr:.4f} -> {current_asr:.4f})"
                elif abs(current_asr - best_asr) < 1e-6 and current_l2 < best_l2:
                    is_new_best = True
                    update_reason = f"ASR тот же, но L2 норма меньше ({best_l2:.4f} -> {current_l2:.4f})"

                if is_new_best:
                    best_config_info = results
                    logger.info(
                        f"*** Найдена новая лучшая конфигурация! "
                        f"Причина: {update_reason} ***"
                    )

                state_to_save = {
                    'exp_config': exp_config,
                    'config_path': config_path,
                    'start_trial': current_trial_num,
                    'all_trials_results': all_trials_results,
                    'best_config_info': best_config_info,
                }
                with open(checkpoint_filepath, 'wb') as f:
                    pickle.dump(state_to_save, f)
                logger.debug(f"Чекпоинт для прогона #{current_trial_num} сохранен.")
        else:
            logger.info("Все прогоны Random Search уже завершены согласно чекпоинту. Пропускаем поиск.")

        logger.info("Поиск фронта Парето из всех случайных прогонов...")

        pareto_front_results = find_pareto_front(all_trials_results)
        logger.info(f"Найдено {len(pareto_front_results)} оптимальных по Парето решений.")

        if pareto_front_results:
            best_in_pareto = max(pareto_front_results, key=lambda r: r.get('attack_success_rate', -1.0))
            if best_in_pareto.get('attack_success_rate', -1.0) > best_config_info.get('attack_success_rate', -1.0):
                best_config_info = best_in_pareto


    elif sampler_type == 'evolutionary':
        logger.info("Запуск поиска с использованием Evolutionary Sampler (генетический алгоритм).")

        evo_config = search_config.get('params', {})
        population_size = evo_config.get('population_size', 50)
        num_generations = exp_config.get('num_trials')

        log_dir = os.path.join(results_dir, f"evolution_logs_{timestamp}")
        os.makedirs(log_dir, exist_ok=True)
        logger.info(f"Логи TensorBoard будут сохраняться в: {log_dir}")

        sampler = EvolutionarySampler(
            search_space_path=exp_config['search_space_path'],
            population_size=population_size,
            mutation_rate=evo_config.get('mutation_rate', 0.1),
            tournament_size=evo_config.get('tournament_size', 3),
            norm=exp_config['norm'],
            objectives=evo_config.get('objectives'),
            constraints=evo_config.get('constraints'),
            mutation_strategy=evo_config.get('mutation_strategy')
        )
        tb_logger = TensorBoardLogger(log_dir)

        if resume_path:
            sampler.__dict__.update(sampler_state)
            best_overall_config = best_config_info
            logger.info(f"Состояние популяции Evolutionary Sampler и лучший результат восстановлены.")
        else:
            sampler.initialize_population()
            best_overall_config = {}

        remaining_generations = num_generations - start_generation
        if remaining_generations > 0:
            logger.info(
                f"Запуск {remaining_generations} поколений эволюции (с {start_generation + 1} по {num_generations}).")

            for gen in range(start_generation, num_generations):
                current_gen_num = gen + 1

                logger.info(f"--- Поколение {current_gen_num}/{num_generations} ---")
                current_population_configs = [ind.copy() for ind in sampler.population]
                for config in current_population_configs:
                    config['epsilon'] = exp_config['epsilon']
                    config['steps'] = exp_config['steps']

                logger.info(f"Оценка {len(current_population_configs)} индивидов...")
                tasks = [
                    delayed(evaluate_config)(
                        conf, model_wrappers, dataloader, device,
                        progress_info=f"Пок. {current_gen_num}, инд. {i + 1}/{len(current_population_configs)}"
                    ) for i, conf in enumerate(current_population_configs)
                ]

                results_list = Parallel(n_jobs=1)(tasks)
                population_results = []
                for res_tuple in results_list:
                    if res_tuple and res_tuple[0]:
                        res_data = res_tuple[0]
                        res_data['generation'] = gen
                        population_results.append(res_data)
                all_trials_results.extend(population_results)

                if not population_results:
                    logger.warning(
                        "Не получено ни одного валидного результата в этом поколении. Пропускаем эволюцию и сохранение.")
                    continue

                objectives_config = evo_config.get('objectives')
                if objectives_config:
                    logger.debug("Вычисление приспособленности в многоцелевом режиме.")
                    fitnesses_for_evolution = []
                    for res in population_results:
                        fitness_tuple = []
                        for obj in objectives_config:
                            metric_val = res.get(obj['metric'],
                                                 -1.0 if obj['direction'] == 'maximize' else float('inf'))
                            if obj['direction'] == 'maximize':
                                metric_val *= -1
                            fitness_tuple.append(metric_val)
                        fitnesses_for_evolution.append(tuple(fitness_tuple))
                else:
                    logger.debug("Вычисление приспособленности в одноцелевом режиме (ASR).")
                    fitnesses_for_evolution = [res.get('attack_success_rate', -1.0) for res in population_results]

                asr_scores_for_logging = [res.get('attack_success_rate', -1.0) for res in population_results]

                tb_logger.log_generation_stats(sampler.population, asr_scores_for_logging, gen)
                tb_logger.log_population_distribution_plot(sampler.population, gen)

                best_asr_in_gen = max(asr_scores_for_logging)
                best_idx_in_gen = asr_scores_for_logging.index(best_asr_in_gen)
                best_config_in_gen = population_results[best_idx_in_gen]

                tb_logger.log_best_individual(best_config_in_gen['processed_config'], best_asr_in_gen, gen)

                if not best_overall_config or best_asr_in_gen > best_overall_config.get('attack_success_rate', -1.0):
                    best_overall_config = best_config_in_gen
                    logger.info(
                        f"*** Найдена новая лучшая конфигурация за все время! "
                        f"ASR: {best_overall_config['attack_success_rate']:.4f} ***"
                    )

                sampler.evolve(fitnesses_for_evolution, population_results)

                state_to_save = {
                    'exp_config': exp_config,
                    'config_path': config_path,
                    'start_generation': gen + 1,
                    'all_trials_results': all_trials_results,
                    'best_config_info': best_overall_config,
                    'sampler_state': sampler.__dict__,
                }
                with open(checkpoint_filepath, 'wb') as f:
                    pickle.dump(state_to_save, f)
                logger.info(f"Чекпоинт для поколения #{current_gen_num} сохранен.")
        else:
            logger.info("Все поколения эволюции уже завершены согласно чекпоинту. Пропускаем поиск.")

        tb_logger.close()
        best_config_info = best_overall_config

        logger.info("Поиск фронта Парето из всех результатов эволюции...")
        pareto_front_results = find_pareto_front(all_trials_results)
        logger.info(f"Найдено {len(pareto_front_results)} оптимальных по Парето решений.")
    else:
        raise ValueError(f"Неизвестный тип сэмплера: '{sampler_type}'. Доступны: 'random', 'optuna'.")

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
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help="(Опционально) Путь к файлу чекпоинта (*.pkl) для возобновления поиска."
    )
    args = parser.parse_args()
    main(args.config, resume_path=args.resume)