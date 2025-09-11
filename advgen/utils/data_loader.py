"""
Этот модуль предоставляет функции для загрузки и предобработки
популярных датасетов для задач компьютерного зрения.
"""

import logging
import os
from typing import Tuple, Dict, Any, Optional

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset

logger = logging.getLogger(__name__)

DATASET_STATS = {
    'cifar10': {
        'mean': [0.4914, 0.4822, 0.4465],
        'std': [0.2023, 0.1994, 0.2010],
    },
    'cifar100': {
        'mean': [0.5071, 0.4867, 0.4408],
        'std': [0.2675, 0.2565, 0.2761],
    },
    'imagenet': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
    }
}


def _get_transform(dataset_name: str) -> transforms.Compose:
    """Возвращает соответствующий transform для датасета."""
    if dataset_name in ['cifar10', 'cifar100']:
        return transforms.Compose([transforms.ToTensor()])
    elif dataset_name == 'imagenet':
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
    else:
        return transforms.Compose([transforms.ToTensor()])


def _get_dataset(dataset_name: str, data_dir: str, transform: transforms.Compose) -> Dataset:
    """Создает и возвращает экземпляр тестового датасета."""
    if dataset_name == 'cifar10':
        return torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform
        )
    elif dataset_name == 'cifar100':
        return torchvision.datasets.CIFAR100(
            root=data_dir, train=False, download=True, transform=transform
        )
    elif dataset_name == 'imagenet':
        val_dir = os.path.join(data_dir, "imagenet", "val")
        if not os.path.isdir(val_dir):
            raise FileNotFoundError(
                f"Директория валидации ImageNet не найдена по пути: {val_dir}. "
                "Убедитесь, что датасет загружен и имеет структуру data/imagenet/val/{class}/{image.JPEG}"
            )
        return torchvision.datasets.ImageFolder(root=val_dir, transform=transform)
    else:
        raise NotImplementedError(f"Логика загрузки для датасета '{dataset_name}' не реализована.")


def get_dataloader(
        dataset_name: str,
        data_dir: str = './data',
        batch_size: int = 128,
        num_samples: Optional[int] = 1000,
        shuffle: bool = False,
        num_workers: int = 4,
        pin_memory: bool = True
) -> Tuple[DataLoader, Dict[str, Any]]:
    """
    Создает и возвращает DataLoader для указанного датасета.

    :param dataset_name: Название датасета. Поддерживаются: 'cifar10', 'cifar100', 'imagenet'.
    :param data_dir: Путь к директории для хранения данных.
    :param batch_size: Размер батча.
    :param num_samples: (Опционально) Количество примеров для использования. Если None, используется весь тестовый датасет.
    :param shuffle: Перемешивать ли данные. Для тестов обычно False.
    :param num_workers: Количество потоков для загрузки данных.
    :param pin_memory: Использовать ли pin_memory для ускорения передачи на GPU.
    :return: Кортеж, содержащий: DataLoader и словарь со статистикой датасета (mean, std).
    """
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_STATS:
        raise ValueError(f"Неизвестное имя датасета: {dataset_name}. Поддерживаются: {list(DATASET_STATS.keys())}")

    stats = DATASET_STATS[dataset_name]
    transform = _get_transform(dataset_name)

    logger.info(f"Загрузка тестового сета для '{dataset_name}'...")
    full_testset = _get_dataset(dataset_name, data_dir, transform)

    if num_samples is not None:
        if num_samples > len(full_testset):
            logger.warning(
                f"Запрошено {num_samples} примеров, но в датасете "
                f"доступно только {len(full_testset)}. Будет использован весь датасет."
            )
            num_samples = len(full_testset)

        generator = torch.Generator().manual_seed(42)  # Для воспроизводимости подвыборки
        subset_indices = torch.randperm(len(full_testset), generator=generator)[:num_samples]
        dataset_to_load = Subset(full_testset, subset_indices)
        logger.info(f"Используется подвыборка из {len(dataset_to_load)} случайных примеров.")
    else:
        dataset_to_load = full_testset
        logger.info(f"Используется полный тестовый датасет из {len(dataset_to_load)} примеров.")

    dataloader = DataLoader(
        dataset_to_load,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    return dataloader, stats