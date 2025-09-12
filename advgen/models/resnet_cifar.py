import torch.nn as nn
import torchvision.models as models


def resnet18_cifar(num_classes=10, pretrained=True):
    """Адаптирует ResNet18 из torchvision для работы с CIFAR-10."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    model.maxpool = nn.Identity()

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model


def resnet18_imagenet_arch(num_classes=10, pretrained=True):
    """
    Создает стандартную архитектуру ResNet18 из torchvision,
    но с измененным последним слоем под нужное количество классов.
    Сохраняет оригинальный conv1 (7x7) и maxpool.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    if model.fc.out_features != num_classes:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    return model