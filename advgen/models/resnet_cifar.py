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