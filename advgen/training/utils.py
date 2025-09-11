"""
Этот модуль содержит вспомогательные функции для процесса обучения,
в частности, для сохранения и загрузки чекпоинтов моделей.
"""

import logging
import os
import shutil
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


def save_checkpoint(
        state: Dict[str, Any],
        is_best: bool,
        path: str,
        filename: str = 'checkpoint.pth.tar'
):
    """
    Сохраняет состояние модели в чекпоинт.

    Если чекпоинт является лучшим на данный момент (is_best=True),
    он также копируется в файл 'model_best.pth.tar'.

    :param state: Словарь с состоянием, который нужно сохранить.
                  Обычно содержит 'epoch', 'state_dict', 'optimizer', 'best_acc'.
    :param is_best: Флаг, указывающий, является ли текущая модель лучшей.
    :param path: Путь к директории для сохранения.
    :param filename: Имя файла для обычного чекпоинта.
    """
    filepath = os.path.join(path, filename)
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(path, 'model_best.pth.tar')
        shutil.copyfile(filepath, best_filepath)
        logger.debug(f"Лучший чекпоинт сохранен в '{best_filepath}'")


def load_checkpoint(
        path: str,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
) -> Dict[str, Any]:
    """
    Загружает чекпоинт модели из файла.

    :param path: Путь к файлу чекпоинта (.pth.tar).
    :param model: Экземпляр модели, в которую будут загружены веса.
    :param optimizer: (Опционально) Экземпляр оптимизатора для загрузки состояния.
    :param scheduler: (Опционально) Экземпляр планировщика для загрузки состояния.
    :return: Словарь с информацией из чекпоинта (например, номер эпохи, лучшая точность).
    :raises FileNotFoundError: Если файл чекпоинта не найден.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Файл чекпоинта не найден по пути: {path}")

    device = next(model.parameters()).device
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint['state_dict'])

    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("Состояние оптимизатора успешно загружено.")

    if scheduler and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
        logger.info("Состояние планировщика успешно загружено.")

    start_epoch = checkpoint.get('epoch', 0)
    best_acc = checkpoint.get('best_acc', 0.0)

    logger.info(f"Чекпоинт успешно загружен из '{path}'.\n"
                f"  - Эпоха: {start_epoch}\n"
                f"  - Лучшая точность: {best_acc:.4f}")

    return {
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'start_epoch': start_epoch,
        'best_acc': best_acc,
    }