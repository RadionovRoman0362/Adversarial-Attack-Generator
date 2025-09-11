"""
Этот модуль служит "фабрикой" для создания компонентов состязательной атаки.

Он содержит:
1. COMPONENT_REGISTRY: Словарь, который сопоставляет строковые имена
   компонентов (используемые в YAML-конфигах) с их классами.
2. create_component: Фабричная функция, которая принимает тип компонента,
   его имя и параметры, находит соответствующий класс в реестре и
   создает его экземпляр.

Такой подход позволяет легко добавлять новые компоненты: достаточно
реализовать класс в соответствующем модуле и зарегистрировать его здесь.
Основной код (AttackRunner) остается неизменным.
"""

from typing import Type, Any, Dict

from .base import (
    Initializer,
    Loss,
    GradientComputer,
    Scheduler,
    Projector,
    UpdateRule
)
from .initializers import (
    ZeroInitializer,
    RandomLinfInitializer,
    RandomL2Initializer
)
from .losses import (
    CrossEntropyLoss,
    UntargetedCWLoss,
    DifferenceOfLogitsRatioLoss,
    PopperLoss
)
from .gradients import (
    StandardGradient,
    MomentumGradient,
    AdversarialGradientSmoothing,
    InputDiversityGradient,
    TranslationInvariantGradient
)
from .schedulers import (
    FixedStepScheduler,
    LinearDecayStepScheduler,
    CosineAnnealingStepScheduler,
    ExponentialDecayStepScheduler,
    MultiStepDecayStepScheduler,
    AdaptiveStepScheduler
)
from .projectors import (
    NoProjection,
    LinfProjector,
    L2Projector,
    L1Projector,
    L0Projector
)

from .updaters import (
    StandardUpdate,
    SignUpdate,
    L2Update,
    AdamUpdate,
    L1MedianUpdate
)

COMPONENT_REGISTRY: Dict[str, Dict[str, Type[Any]]] = {
    "initializer": {
        "zero": ZeroInitializer,
        "random_linf": RandomLinfInitializer,
        "random_l2": RandomL2Initializer,
    },
    "loss": {
        "cross_entropy": CrossEntropyLoss,
        "cw": UntargetedCWLoss,
        "dlr": DifferenceOfLogitsRatioLoss,
        "popper": PopperLoss,
    },
    "gradient": {
        "standard": StandardGradient,
        "momentum": MomentumGradient,
        "smooth_adv": AdversarialGradientSmoothing,
        "di": InputDiversityGradient,
        "ti": TranslationInvariantGradient,
    },
    "scheduler": {
        "fixed": FixedStepScheduler,
        "linear": LinearDecayStepScheduler,
        "cosine": CosineAnnealingStepScheduler,
        "exponential": ExponentialDecayStepScheduler,
        "multistep": MultiStepDecayStepScheduler,
        "adaptive": AdaptiveStepScheduler,
    },
    "projector": {
        "none": NoProjection,
        "linf": LinfProjector,
        "l2": L2Projector,
        "l1": L1Projector,
        "l0": L0Projector,
    },
    "updater": {
        "standard": StandardUpdate,
        "sign": SignUpdate,
        "l2": L2Update,
        "adam": AdamUpdate,
        "l1_median": L1MedianUpdate,
    },
}


def create_component(component_type: str, name: str, params: Dict[str, Any] = None) -> Any:
    """
    Фабричная функция для создания экземпляров компонентов.

    Находит класс компонента в реестре по его типу и имени, а затем
    создает его экземпляр, передавая в конструктор параметры.

    :param component_type: Тип компонента (ключ верхнего уровня в реестре,
                           например, 'loss', 'scheduler').
    :param name: Имя конкретной реализации (ключ второго уровня,
                 например, 'cross_entropy', 'fixed').
    :param params: Словарь с параметрами для конструктора класса.
                   Если None, компонент создается без параметров.
    :return: Экземпляр запрошенного компонента.
    :raises ValueError: Если компонент с указанным типом или именем не найден.
    """
    if params is None:
        params = {}

    try:
        component_cls = COMPONENT_REGISTRY[component_type][name]
    except KeyError:
        raise ValueError(
            f"Неизвестный компонент: тип='{component_type}', имя='{name}'.\n"
            f"Доступные типы: {list(COMPONENT_REGISTRY.keys())}.\n"
            f"Для типа '{component_type}' доступны имена: {list(COMPONENT_REGISTRY.get(component_type, {}).keys())}."
        )

    try:
        return component_cls(**params)
    except TypeError as e:
        raise TypeError(
            f"Ошибка при создании компонента '{name}' типа '{component_type}' "
            f"с параметрами {params}.\n"
            f"Проверьте, что все необходимые параметры для конструктора "
            f"'{component_cls.__name__}' переданы.\n"
            f"Оригинальная ошибка: {e}"
        )


__all__ = [
    "create_component",
    "COMPONENT_REGISTRY",
    "Initializer",
    "Loss",
    "GradientComputer",
    "Scheduler",
    "Projector",
    "UpdateRule"
]