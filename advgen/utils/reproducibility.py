"""
Модуль для обеспечения воспроизводимости экспериментов.
"""
import random
import numpy as np
import torch
import os


def set_seed(seed: int):
    """
    Устанавливает random seed для всех основных библиотек для обеспечения
    воспроизводимости результатов.

    :param seed: Целое число для инициализации генераторов случайных чисел.
    """
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)