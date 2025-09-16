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
from .training_losses import TRADESLoss

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
            dataset_stats: Optional[Dict[str, list]] = None,
            val_attack_runner: Optional[AttackRunner] = None
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
        self.val_attack_runner = val_attack_runner
        if self.attack_runner:
            logger.info("Тренер запущен в режиме состязательного обучения.")

        self.is_trades = isinstance(self.criterion, TRADESLoss)
        if self.is_trades:
            logger.info("Тренер использует функцию потерь TRADES.")

        self.train_model_wrapper = ModelWrapper(
            self.model,
            mean=self.dataset_stats['mean'],
            std=self.dataset_stats['std']
        )
        self.val_model_wrapper = ModelWrapper(
            self.model,
            mean=self.dataset_stats['mean'],
            std=self.dataset_stats['std']
        )
        self.validation_criterion = nn.CrossEntropyLoss()

        self.best_acc = 0.0
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        logger.info(f"Тренер инициализирован. Устройство: {self.device}. "
                    f"Чекпоинты будут сохраняться в '{self.checkpoint_dir}'.")

    def train(self, epochs: int, patience: int = 5):
        """
        Запускает полный цикл обучения на заданное количество эпох.
        """
        logger.info(f"Начало обучения на {epochs} эпох.")

        best_robust_acc = 0.0
        best_clean_acc = 0.0
        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._train_one_epoch(epoch)
            val_metrics = self._validate_one_epoch()

            self.scheduler.step()

            logger.info(
                f"Эпоха {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_metrics.get('val_loss'):.4f}, Val Acc: {val_metrics.get('val_acc'):.4f} | "
                f"Robust Val Loss: {val_metrics.get('robust_loss', 0):.4f}, "
                f"Robust Val Acc: {val_metrics.get('robust_acc', 0):.4f}"
            )

            current_robust_acc = val_metrics.get('robust_acc', val_metrics['val_acc'])
            current_clean_acc = val_metrics.get('val_acc')

            is_best = (current_robust_acc > best_robust_acc) or (current_clean_acc > best_clean_acc)
            if is_best:
                if current_robust_acc > best_robust_acc:
                    best_robust_acc = current_robust_acc
                if current_clean_acc > best_clean_acc:
                    best_clean_acc = current_clean_acc
                self.best_acc = val_metrics.get('val_acc')
                epochs_without_improvement = 0
                logger.info(f"*** Новая лучшая точность на валидации: {self.best_acc:.4f}! "
                            f"Сохранение модели... ***")
            else:
                epochs_without_improvement += 1

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

            if epochs_without_improvement >= patience:
                logger.info(f"Обучение остановлено досрочно: робастная точность не улучшалась {patience} эпох.")
                break

        logger.info(f"Обучение завершено. Лучшая точность на валидации: {self.best_acc:.4f}")

    def _train_one_epoch(self, epoch_num: int) -> tuple[float, float]:
        """Выполняет один проход обучения по всему обучающему датасету."""
        self.train_model_wrapper.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        outputs_clean = None
        outputs_adv = None

        pbar = tqdm(self.train_loader, desc=f"Эпоха {epoch_num} (Train)", dynamic_ncols=True)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            if not self.attack_runner or self.is_trades:
                outputs_clean = self.train_model_wrapper(inputs)

            if self.attack_runner:
                adv_inputs = self.attack_runner.attack(
                    self.train_model_wrapper,
                    inputs,
                    labels,
                    keep_graph=True
                )
                outputs_adv = self.train_model_wrapper(adv_inputs)

            if self.is_trades:
                loss = self.criterion(outputs_clean, outputs_adv, labels)
                outputs_for_acc = outputs_adv
            elif self.attack_runner:
                loss = self.criterion(outputs_adv, labels)
                outputs_for_acc = outputs_adv
            else:
                loss = self.criterion(outputs_clean, labels)
                outputs_for_acc = outputs_clean

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs_for_acc.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            pbar.set_postfix(Loss=f"{total_loss / total_samples:.4f}",
                             Acc=f"{correct_predictions / total_samples:.4f}")

        avg_loss = total_loss / total_samples
        avg_acc = correct_predictions / total_samples
        return avg_loss, avg_acc

    def _validate_one_epoch(self) -> Dict[str, float]:
        """Выполняет один проход валидации по всему валидационному датасету."""
        self.val_model_wrapper.eval()
        metrics = {}
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        pbar = tqdm(self.val_loader, desc="Валидация", leave=False, dynamic_ncols=True)
        with torch.no_grad():
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.val_model_wrapper(inputs)
                loss = self.validation_criterion(outputs, labels)

                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

                pbar.set_postfix(Loss=f"{total_loss / total_samples:.4f}",
                                 Acc=f"{correct_predictions / total_samples:.4f}")

        clean_loss = total_loss / total_samples
        clean_acc = correct_predictions / total_samples

        metrics['val_loss'] = clean_loss
        metrics['val_acc'] = clean_acc

        if self.val_attack_runner:
            total_loss_adv = 0.0
            correct_predictions_adv = 0
            total_samples_adv = 0

            pbar_adv = tqdm(self.val_loader, desc="Робастная Валидация", leave=False, dynamic_ncols=True)

            for inputs, labels in pbar_adv:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                adv_inputs = self.val_attack_runner.attack(self.val_model_wrapper, inputs, labels)

                with torch.no_grad():
                    outputs = self.val_model_wrapper(adv_inputs)
                    loss = self.validation_criterion(outputs, labels)

                total_loss_adv += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions_adv += (predicted == labels).sum().item()
                total_samples_adv += labels.size(0)

                pbar_adv.set_postfix(Loss=f"{total_loss_adv / total_samples_adv:.4f}",
                                     Acc=f"{correct_predictions_adv / total_samples_adv:.4f}")

            robust_loss = total_loss_adv / total_samples_adv
            robust_acc = correct_predictions_adv / total_samples_adv

            metrics['robust_loss'] = robust_loss
            metrics['robust_acc'] = robust_acc

        return metrics