import math
import torch
import torch.nn.functional as F


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp_min(1.0)
    return ((pred - target) ** 2 * mask).sum() / denom


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp_min(1.0)
    return ((pred - target).abs() * mask).sum() / denom


def weighted_masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    intensity_weight: float = 8.0,
    intensity_gamma: float = 1.0,
) -> torch.Tensor:
    """Masked MSE with higher weight on bright diffraction-spot pixels.

    The denominator is the sum of weights inside the mask, not the number of
    pixels. This prevents large circular masks from being dominated by dark
    background pixels around a diffraction spot.
    """
    gamma = max(float(intensity_gamma), 0.0)
    weight = 1.0 + float(intensity_weight) * target.clamp_min(0.0).pow(gamma)
    weighted_mask = mask * weight
    denom = weighted_mask.sum().clamp_min(1.0)
    return ((pred - target) ** 2 * weighted_mask).sum() / denom


def psnr_from_mse(mse: float, max_value: float = 1.0) -> float:
    if mse <= 0:
        return float("inf")
    return 20.0 * math.log10(max_value) - 10.0 * math.log10(mse)


def simple_ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = F.avg_pool2d(pred, 7, stride=1, padding=3)
    mu_y = F.avg_pool2d(target, 7, stride=1, padding=3)
    sigma_x = F.avg_pool2d(pred * pred, 7, stride=1, padding=3) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, 7, stride=1, padding=3) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, 7, stride=1, padding=3) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2))
    return ssim.mean()
