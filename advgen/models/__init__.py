import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any

from . import resnet_cifar  # Убедитесь, что импортируете ваш кастомный resnet


def get_model(config: Dict[str, Any]) -> nn.Module:
    """
    Центральная фабричная функция для создания моделей.
    """
    model_name = config['model_name']
    num_classes = config.get('num_classes', 10)
    pretrained = config.get('pretrained', False)

    if model_name == 'resnet18_cifar':
        return resnet_cifar.resnet18_cifar(num_classes=num_classes, pretrained=pretrained)

    elif model_name == 'resnet18_imagenet':
        return resnet_cifar.resnet18_imagenet_arch(num_classes, pretrained)

    else:
        raise NotImplementedError(f"Модель '{model_name}' не поддерживается.")


__all__ = ["get_model", "resnet_cifar"]