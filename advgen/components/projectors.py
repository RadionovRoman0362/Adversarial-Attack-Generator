"""
Этот модуль содержит конкретные реализации проекторов.
Проектор отвечает за то, чтобы состязательное возмущение оставалось
в пределах допустимой эпсилон-окрестности от оригинального изображения
согласно выбранной метрике (норме), а также в валидном диапазоне значений
пикселей [0, 1].

Все классы наследуются от `Projector` из `base.py`.
"""

import torch

from .base import Projector


class NoProjection(Projector):
    """
    "Пустой" проектор, который не ограничивает норму возмущения.
    Осуществляет только отсечение значений пикселей в валидный диапазон [0, 1].
    Полезен для отладки и исследования "неограниченных" атак.
    """

    def project(
            self,
            adv_images: torch.Tensor,
            original_images: torch.Tensor,
            epsilon: float
    ) -> torch.Tensor:
        """
        :param adv_images: Текущие состязательные изображения после шага градиента.
        :param original_images: Оригинальные изображения.
        :param epsilon: Параметр не используется.
        """
        return torch.clamp(adv_images, min=0, max=1)


class LinfProjector(Projector):
    """
    Проектор для L-infinity (L∞) нормы.
    Гарантирует, что максимальное абсолютное изменение любого пикселя
    не превышает эпсилон.
    """

    def project(
            self,
            adv_images: torch.Tensor,
            original_images: torch.Tensor,
            epsilon: float
    ) -> torch.Tensor:
        """
        Проецирует возмущение на L-inf шар (гиперкуб).
        """
        perturbation = adv_images - original_images
        clipped_perturbation = torch.clamp(perturbation, -epsilon, epsilon)
        projected_images = torch.clamp(original_images + clipped_perturbation, min=0, max=1)
        return projected_images


class L2Projector(Projector):
    """
    Проектор для L2 (евклидовой) нормы.
    Гарантирует, что евклидова норма вектора возмущения не превышает эпсилон.
    """

    def project(
            self,
            adv_images: torch.Tensor,
            original_images: torch.Tensor,
            epsilon: float
    ) -> torch.Tensor:
        """
        Проецирует возмущение на L2 шар.
        """
        perturbation = adv_images - original_images
        batch_size = perturbation.shape[0]

        flat_perturbation = perturbation.view(batch_size, -1)
        norms = torch.linalg.norm(flat_perturbation, ord=2, dim=1)

        scale_factors = epsilon / torch.clamp(norms, min=1e-12)
        scale_factors = torch.min(scale_factors, torch.ones_like(scale_factors))

        clipped_perturbation = perturbation * scale_factors.view(-1, 1, 1, 1)

        projected_images = torch.clamp(original_images + clipped_perturbation, min=0, max=1)
        return projected_images


class L1Projector(Projector):
    """
    Проектор для L1 ("манхэттенской") нормы.
    Сложная проекция, поощряющая разреженные возмущения.
    Реализация основана на алгоритме из статьи Duchi et al., 2008.
    """

    def project(
            self,
            adv_images: torch.Tensor,
            original_images: torch.Tensor,
            epsilon: float
    ) -> torch.Tensor:
        """
        Проецирует возмущение на L1 шар.
        """
        perturbation = adv_images - original_images
        batch_size = perturbation.shape[0]
        flat_perturbation = perturbation.view(batch_size, -1)

        l1_norms = torch.linalg.norm(flat_perturbation, ord=1, dim=1)
        mask_to_project = l1_norms > epsilon

        if not torch.any(mask_to_project):
            return torch.clamp(adv_images, min=0, max=1)

        x_to_project = flat_perturbation[mask_to_project]

        x_abs = torch.abs(x_to_project)
        sorted_x, _ = torch.sort(x_abs, dim=1, descending=True)

        cumsum = torch.cumsum(sorted_x, dim=1)
        n_features = x_to_project.shape[1]
        k_values = (cumsum - epsilon) / (torch.arange(n_features, device=x_to_project.device) + 1.0)

        rho_tensor = torch.where(sorted_x > k_values, torch.arange(n_features, device=x_to_project.device), -1)
        rho, _ = torch.max(rho_tensor, dim=1)

        theta = torch.gather(k_values, 1, rho.unsqueeze(1)).squeeze(1)

        clipped_x = torch.sign(x_to_project) * torch.relu(torch.abs(x_to_project) - theta.unsqueeze(1))

        flat_perturbation[mask_to_project] = clipped_x

        clipped_perturbation = flat_perturbation.view_as(perturbation)
        projected_images = torch.clamp(original_images + clipped_perturbation, min=0, max=1)
        return projected_images


class L0Projector(Projector):
    """
    Проектор для L0 псевдо-нормы.
    Ограничивает количество изменяемых пикселей.
    """

    def project(
            self,
            adv_images: torch.Tensor,
            original_images: torch.Tensor,
            epsilon: float
    ) -> torch.Tensor:
        """
        Оставляет только `epsilon` наибольших по модулю изменений.
        `epsilon` здесь интерпретируется как целое число.
        """
        try:
            k = int(epsilon)
        except (ValueError, TypeError):
            raise TypeError("Для L0Projector параметр epsilon должен быть целым числом.")

        if k == 0:
            return original_images.clone()

        perturbation = adv_images - original_images
        batch_size = perturbation.shape[0]

        n_features = perturbation.numel() // batch_size
        if k > n_features:
            raise ValueError(f"Количество пикселей k={k} не может превышать их общее число {n_features}")

        flat_perturbation = perturbation.view(batch_size, -1)
        abs_perturbation = torch.abs(flat_perturbation)

        top_k_thresholds, _ = torch.kthvalue(abs_perturbation, n_features - k, dim=1, keepdim=True)

        mask = (abs_perturbation >= top_k_thresholds).float()

        clipped_perturbation = (flat_perturbation * mask).view_as(perturbation)

        projected_images = torch.clamp(original_images + clipped_perturbation, min=0, max=1)
        return projected_images