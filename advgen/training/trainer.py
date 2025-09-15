"""
Этот модуль содержит класс Trainer, который инкапсулирует полный цикл
обучения и валидации модели PyTorch.
"""

import logging
import os
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import save_checkpoint
from ..core.attack_runner import AttackRunner
from ..core.model_wrapper import ModelWrapper

logger = logging.getLogger(__name__)


class Trainer:
    """
    Класс для управления процессом обучения и валидации нейронной сети.
    """

    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            optimizer: optim.Optimizer,
            scheduler: optim.lr_scheduler._LRScheduler,
            criterion: nn.Module,
            device: torch.device,
            checkpoint_dir: str,
            attack_runner: Optional[AttackRunner] = None,
            dataset_stats: Optional[Dict[str, list]] = None
    ):
        """
        :param model: Модель PyTorch для обучения.
        :param train_loader: DataLoader для обучающего набора данных.
        :param val_loader: DataLoader для валидационного набора данных.
        :param optimizer: Оптимизатор (например, SGD, Adam).
        :param scheduler: Планировщик скорости обучения.
        :param criterion: Функция потерь (например, CrossEntropyLoss).
        :param device: Устройство для вычислений ('cpu' или 'cuda').
        :param checkpoint_dir: Директория для сохранения чекпоинтов.
        :param attack_runner: Экземпляр AttackRunner для состязательного обучения.
        :param dataset_stats: Статистика для нормализации.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.attack_runner = attack_runner
        self.dataset_stats = dataset_stats
        if self.attack_runner:
            logger.info("Тренер запущен в режиме состязательного обучения.")

        self.best_acc = 0.0
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        logger.info(f"Тренер инициализирован. Устройство: {self.device}. "
                    f"Чекпоинты будут сохраняться в '{self.checkpoint_dir}'.")

    def train(self, epochs: int):
        """
        Запускает полный цикл обучения на заданное количество эпох.
        """
        logger.info(f"Начало обучения на {epochs} эпох.")
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._train_one_epoch(epoch)

            val_loss, val_acc = self._validate_one_epoch()

            self.scheduler.step()

            logger.info(
                f"Эпоха {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

            is_best = val_acc > self.best_acc
            if is_best:
                self.best_acc = val_acc
                logger.info(f"*** Новая лучшая точность на валидации: {self.best_acc:.4f}! "
                            f"Сохранение модели... ***")

            save_checkpoint(
                state={
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'best_acc': self.best_acc,
                },
                is_best=is_best,
                path=self.checkpoint_dir
            )
        logger.info(f"Обучение завершено. Лучшая точность на валидации: {self.best_acc:.4f}")

    def _train_one_epoch(self, epoch_num: int) -> tuple[float, float]:
        """Выполняет один проход обучения по всему обучающему датасету."""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        pbar = tqdm(self.train_loader, desc=f"Эпоха {epoch_num} (Train)", dynamic_ncols=True)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            if self.attack_runner:
                train_model_wrapper = ModelWrapper(
                    self.model,
                    mean=self.dataset_stats['mean'],
                    std=self.dataset_stats['std']
                )

                adv_inputs = self.attack_runner.attack(
                    train_model_wrapper,
                    inputs,
                    labels,
                    keep_graph=True
                )
                outputs = self.model(adv_inputs)
            else:
                outputs = self.model(inputs)

            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            pbar.set_postfix(Loss=f"{total_loss / total_samples:.4f}",
                             Acc=f"{correct_predictions / total_samples:.4f}")

        avg_loss = total_loss / total_samples
        avg_acc = correct_predictions / total_samples
        return avg_loss, avg_acc

    def _validate_one_epoch(self) -> tuple[float, float]:
        """Выполняет один проход валидации по всему валидационному датасету."""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        pbar = tqdm(self.val_loader, desc="Валидация", leave=False, dynamic_ncols=True)
        with torch.no_grad():
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

                pbar.set_postfix(Loss=f"{total_loss / total_samples:.4f}",
                                 Acc=f"{correct_predictions / total_samples:.4f}")

        avg_loss = total_loss / total_samples
        avg_acc = correct_predictions / total_samples
        return avg_loss, avg_acc