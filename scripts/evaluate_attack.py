import torch

from advgen.core.model_wrapper import ModelWrapper

if __name__ == '__main__':
    import torchvision.models as models

    # 1. Загружаем предобученную модель
    resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # 2. Определяем стандартные параметры нормализации для ImageNet
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    # 3. Создаем экземпляр нашей обертки
    wrapped_model = ModelWrapper(model=resnet18, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    print("Создана обертка:")
    print(wrapped_model)

    # 4. Перемещаем на GPU, если доступно
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapped_model.to(device)
    print(f"\nМодель перемещена на устройство: {device}")

    # 5. Создаем "чистый" случайный тензор, имитирующий батч изображений [0, 1]
    dummy_images = torch.rand(4, 3, 224, 224, device=device)
    print(f"Форма входного тензора: {dummy_images.shape}")

    # 6. Получаем логиты через наш стандартизированный интерфейс
    # Внутри ModelWrapper выполнит нормализацию и вызовет resnet18
    with torch.no_grad():  # Отключаем расчет градиентов для простого инференса
        logits = wrapped_model.get_logits(dummy_images)

    print(f"Форма выходных логитов: {logits.shape}")

    # Проверка, что модель в режиме eval
    assert not wrapped_model.model.training
    print("Проверка: Модель находится в режиме .eval()")