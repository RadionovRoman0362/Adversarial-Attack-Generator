"""
Главный скрипт для запуска процесса обучения модели.

Пример запуска из корневой директории проекта:
> python scripts/train.py --config configs/training/train_resnet18_cifar10.yaml
"""

import argparse
import logging
import sys
import os
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from advgen.models import get_model
from advgen.training.trainer import Trainer
from advgen.training.utils import load_checkpoint
from advgen.utils.data_loader import get_dataloader
from advgen.utils.logging_setup import setup_logging
from advgen.core.attack_runner import AttackRunner
from advgen.training.training_losses import TRADESLoss


def get_criterion(config: Dict[str, Any]) -> nn.Module:
    """Фабричная функция для создания функции потерь."""
    criterion_config = config.get('criterion', {'name': 'cross_entropy'})
    name = criterion_config['name']
    params = criterion_config.get('params', {})

    if name == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif name == 'trades':
        logger = logging.getLogger(__name__)
        logger.info(f"Используется TRADESLoss с параметрами: {params}")
        return TRADESLoss(**params)
    else:
        raise NotImplementedError(f"Функция потерь '{name}' не поддерживается.")


def get_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """Фабричная функция для создания оптимизаторов."""
    optimizer_config = config['optimizer']
    name = optimizer_config['name']
    params = optimizer_config.get('params', {})

    if name == 'sgd':
        return optim.SGD(model.parameters(), **params)
    elif name == 'adam':
        return optim.Adam(model.parameters(), **params)
    else:
        raise NotImplementedError(f"Оптимизатор '{name}' не поддерживается.")


def get_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any]) -> optim.lr_scheduler._LRScheduler:
    """Фабричная функция для создания планировщиков."""
    scheduler_config = config['scheduler']
    name = scheduler_config['name']
    params = scheduler_config.get('params', {})

    if name == 'steplr':
        return optim.lr_scheduler.StepLR(optimizer, **params)
    elif name == 'multisteplr':
        return optim.lr_scheduler.MultiStepLR(optimizer, **params)
    elif name == 'multisteplr':
        return optim.lr_scheduler.MultiStepLR(optimizer, **params)
    elif name == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, **params)
    else:
        raise NotImplementedError(f"Планировщик '{name}' не поддерживается.")


def main(config_path: str, resume_path: Optional[str] = None):
    """
    Основная функция для запуска обучения.

    :param config_path: Путь к YAML-конфигу для обучения.
    :param resume_path: (Опционально) Путь к чекпоинту для возобновления обучения.
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info(f"Загрузка конфигурации из: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    logger.info(f"Используемое устройство: {device}")

    logger.info("Подготовка загрузчиков данных...")
    train_loader, stats = get_dataloader(
        dataset_name=config['dataset_name'],
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        num_samples=None,
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        train=True
    )
    val_loader, _ = get_dataloader(
        dataset_name=config['dataset_name'],
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        num_samples=None,
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        train=False
    )

    attack_runner = None
    adv_train_config = config.get('adversarial_training')
    if adv_train_config and adv_train_config.get('enabled'):
        logger.info("Активировано состязательное обучение. Создание AttackRunner...")
        attack_config = adv_train_config['attack_config']
        attack_runner = AttackRunner(attack_config)

    logger.info(f"Создание модели: {config['model_name']}")
    model = get_model(config).to(device)

    logger.info(f"Создание оптимизатора: {config['optimizer']['name']}")
    optimizer = get_optimizer(model, config)

    logger.info(f"Создание планировщика: {config['scheduler']['name']}")
    scheduler = get_scheduler(optimizer, config)

    criterion = get_criterion(config)

    if resume_path:
        logger.info(f"Возобновление обучения с чекпоинта: {resume_path}")
        try:
            checkpoint_data = load_checkpoint(resume_path, model, optimizer, scheduler)
            start_epoch = checkpoint_data['start_epoch']
            best_acc = checkpoint_data['best_acc']
        except FileNotFoundError as e:
            logger.error(e)
            return
    else:
        start_epoch = 0
        best_acc = 0.0

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        checkpoint_dir=config['checkpoint_dir'],
        attack_runner=attack_runner,
        dataset_stats=stats,
        val_attack_runner=attack_runner
    )
    trainer.best_acc = best_acc

    num_epochs = config['epochs']
    logger.info(f"Запуск обучения с эпохи {start_epoch + 1} до {num_epochs}")
    trainer.train(epochs=num_epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Скрипт для обучения моделей.")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help="Путь к YAML-файлу с конфигурацией обучения."
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help="(Опционально) Путь к файлу чекпоинта для возобновления обучения."
    )
    args = parser.parse_args()

    main(args.config, args.resume)