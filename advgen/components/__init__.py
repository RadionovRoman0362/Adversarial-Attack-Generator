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
    TranslationInvariantGradient,
    EnsembleGradient,
    SGM,
    TAP
)
from .schedulers import (
    FixedStepScheduler,
    LinearDecayStepScheduler,
    CosineAnnealingStepScheduler,
    ExponentialDecayStepScheduler,
    MultiStepDecayStepScheduler,
    AdaptiveStepScheduler,
    WarmupCosineDecayScheduler,
    CyclicLRStepScheduler,
    PlateauReduceStepScheduler
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
        "ensemble": EnsembleGradient,
        "sgm": SGM,
        "tap": TAP,
    },
    "scheduler": {
        "fixed": FixedStepScheduler,
        "linear": LinearDecayStepScheduler,
        "cosine": CosineAnnealingStepScheduler,
        "exponential": ExponentialDecayStepScheduler,
        "multistep": MultiStepDecayStepScheduler,
        "adaptive": AdaptiveStepScheduler,
        "warmup_cosine": WarmupCosineDecayScheduler,
        "cyclic": CyclicLRStepScheduler,
        "plateau": PlateauReduceStepScheduler,
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


def create_component(component_type: str, config: Dict[str, Any]) -> Any:
    """
    Фабричная функция для создания экземпляров компонентов с поддержкой
    рекурсивной композиции (паттерн "Декоратор").

    Находит класс компонента в реестре по его типу и имени. Если в конфигурации
    присутствует ключ 'wrapped', функция рекурсивно вызывает саму себя для
    создания вложенного компонента, а затем передает его в конструктор
    компонента-декоратора.

    :param component_type: Тип компонента (ключ верхнего уровня в реестре,
                           например, 'gradient').
    :param config: Словарь с конфигурацией для компонента.
                   Пример: {'name': 'momentum', 'params': {...}, 'wrapped': {'name': 'standard'}}
    :return: Экземпляр запрошенного компонента.
    :raises ValueError: Если компонент с указанным типом или именем не найден.
    :raises TypeError: Если переданы некорректные параметры в конструктор.
    """
    if not config:
        return None

    name = config.get('name')
    if not name:
        raise ValueError(f"В конфигурации компонента типа '{component_type}' отсутствует обязательный ключ 'name'.")

    params = config.get('params', {}).copy()

    try:
        component_cls = COMPONENT_REGISTRY[component_type][name]
    except KeyError:
        raise ValueError(
            f"Неизвестный компонент: тип='{component_type}', имя='{name}'.\n"
            f"Доступные типы: {list(COMPONENT_REGISTRY.keys())}.\n"
            f"Для типа '{component_type}' доступны имена: {list(COMPONENT_REGISTRY.get(component_type, {}).keys())}."
        )

    if 'wrapped' in config:
        wrapped_config = config['wrapped']
        wrapped_component = create_component(component_type, wrapped_config)

        if component_type == 'gradient':
            params['wrapped_computer'] = wrapped_component
        else:
            pass

    try:
        return component_cls(**params)
    except TypeError as e:
        import inspect
        sig = inspect.signature(component_cls.__init__)
        expected_params = list(sig.parameters.keys())
        expected_params.remove('self')

        raise TypeError(
            f"Ошибка при создании компонента '{name}' типа '{component_type}' "
            f"с параметрами {list(params.keys())}.\n"
            f"Конструктор '{component_cls.__name__}' ожидает параметры: {expected_params}.\n"
            f"Проверьте, что все необходимые параметры (включая 'wrapped_computer' для декораторов) переданы.\n"
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