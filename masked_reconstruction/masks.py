import math
from typing import Optional

import torch
import torch.nn.functional as F


def random_patch_mask(batch_size: int, height: int, width: int, patch_size: int, mask_ratio: float, device=None) -> torch.Tensor:
    patch_size = max(1, int(patch_size))
    gh = math.ceil(height / patch_size)
    gw = math.ceil(width / patch_size)
    num_patches = gh * gw
    num_mask = max(1, int(round(mask_ratio * num_patches)))
    masks = torch.zeros((batch_size, gh * gw), device=device, dtype=torch.float32)
    for b in range(batch_size):
        perm = torch.randperm(num_patches, device=device)[:num_mask]
        masks[b, perm] = 1.0
    masks = masks.view(batch_size, 1, gh, gw)
    masks = F.interpolate(masks, size=(height, width), mode="nearest")
    return masks


def wedge_mask(batch_size: int, height: int, width: int, mask_ratio: float, device=None) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    angles = torch.atan2(yy.float() - cy, xx.float() - cx)
    sector_width = max(0.01, min(1.0, float(mask_ratio))) * 2.0 * math.pi
    masks = []
    for _ in range(batch_size):
        start = torch.rand((), device=device) * 2.0 * math.pi - math.pi
        delta = (angles - start + math.pi) % (2.0 * math.pi) - math.pi
        masks.append((delta.abs() <= sector_width / 2.0).float())
    return torch.stack(masks, dim=0).unsqueeze(1)


def central_disk_mask(batch_size: int, height: int, width: int, mask_ratio: float, device=None) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    radius = math.sqrt(max(0.001, min(1.0, float(mask_ratio))) * height * width / math.pi)
    mask = (((yy.float() - cy) ** 2 + (xx.float() - cx) ** 2) <= radius ** 2).float()
    return mask.expand(batch_size, 1, height, width).clone()


def _scaled_radius(value: float, height: int, width: int) -> int:
    if value <= 0:
        raise ValueError("Spot mask radius values must be positive")
    if value < 1:
        return max(1, int(round(value * min(height, width))))
    return max(1, int(round(value)))


def _find_center_beam(pattern: torch.Tensor, valid: torch.Tensor) -> tuple[int, int]:
    score = pattern.float().clone()
    score[~valid.bool()] = -float("inf")
    flat_index = int(torch.argmax(score).item())
    return divmod(flat_index, pattern.shape[1])


def _spot_candidates(
    pattern: torch.Tensor,
    valid: torch.Tensor,
    center_exclude_radius: int,
    peak_window: int,
    min_peak_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    height, width = pattern.shape
    peak_window = max(3, int(peak_window) | 1)
    pad = peak_window // 2
    smoothed = F.avg_pool2d(pattern[None, None].float(), kernel_size=3, stride=1, padding=1)[0, 0]
    local_max = F.max_pool2d(smoothed[None, None], kernel_size=peak_window, stride=1, padding=pad)[0, 0]

    center_y, center_x = _find_center_beam(smoothed, valid)
    yy, xx = torch.meshgrid(torch.arange(height, device=pattern.device), torch.arange(width, device=pattern.device), indexing="ij")
    distance = torch.sqrt((yy.float() - center_y) ** 2 + (xx.float() - center_x) ** 2)
    outside_center = distance > float(center_exclude_radius)
    valid = valid.bool() & outside_center & torch.isfinite(smoothed)
    candidates = (smoothed == local_max) & valid

    scores = smoothed[candidates]
    candidate_yx = torch.nonzero(candidates, as_tuple=False)
    candidate_dist = distance[candidates]
    if scores.numel() == 0:
        fallback = smoothed.clone()
        fallback[~valid] = -float("inf")
        flat_index = int(torch.argmax(fallback).item())
        y, x = divmod(flat_index, width)
        return torch.tensor([[y, x]], device=pattern.device), torch.tensor([smoothed[y, x]], device=pattern.device), torch.tensor([distance[y, x]], device=pattern.device)

    cutoff = scores.max() * max(0.0, min(1.0, float(min_peak_fraction)))
    keep = scores >= cutoff
    if keep.any():
        candidate_yx = candidate_yx[keep]
        scores = scores[keep]
        candidate_dist = candidate_dist[keep]
    return candidate_yx, scores, candidate_dist


def _select_spot_indices(scores: torch.Tensor, distances: torch.Tensor, count: int, spot_selection: str) -> torch.Tensor:
    count = min(max(1, int(count)), int(scores.numel()))
    selection = spot_selection.lower()
    if selection in {"random", "random_spots"}:
        return torch.randperm(scores.numel(), device=scores.device)[:count]
    if selection in {"nearest", "nearest_bright", "near_center"}:
        order = torch.argsort(distances + 1e-3 * (scores.max() - scores))
        return order[:count]
    order = torch.argsort(scores, descending=True)
    return order[:count]


def spot_mask(
    images: torch.Tensor,
    spot_radius: float = 0.05,
    center_exclude_radius: float = 0.06,
    spot_rank: int = 1,
    peak_window: int = 11,
    valid_mask: Optional[torch.Tensor] = None,
    spot_selection: str = "random",
    min_spots: int = 1,
    max_spots: int = 4,
    min_peak_fraction: float = 0.05,
) -> torch.Tensor:
    if images.ndim == 3:
        images = images.unsqueeze(1)
    if images.ndim != 4:
        raise ValueError(f"spot_mask expects images with shape (B,C,H,W), got {tuple(images.shape)}")

    batch_size, _, height, width = images.shape
    spot_radius_px = _scaled_radius(float(spot_radius), height, width)
    center_exclude_px = _scaled_radius(float(center_exclude_radius), height, width)
    yy, xx = torch.meshgrid(torch.arange(height, device=images.device), torch.arange(width, device=images.device), indexing="ij")

    if valid_mask is None:
        valid_mask = torch.ones((batch_size, 1, height, width), device=images.device, dtype=torch.bool)
    elif valid_mask.ndim == 3:
        valid_mask = valid_mask.unsqueeze(1)

    masks = []
    min_spots = max(1, int(min_spots))
    max_spots = max(min_spots, int(max_spots))
    for b in range(batch_size):
        pattern = images[b, 0]
        valid = valid_mask[b, 0] > 0
        candidate_yx, scores, distances = _spot_candidates(pattern, valid, center_exclude_px, peak_window, min_peak_fraction)
        available = int(candidate_yx.shape[0])
        if spot_selection.lower() in {"random", "random_spots"}:
            upper = min(max_spots, available)
            lower = min(min_spots, upper)
            count = int(torch.randint(lower, upper + 1, (), device=images.device).item())
        else:
            count = min(max(1, int(spot_rank)), available)
        selected = _select_spot_indices(scores, distances, count, spot_selection)

        mask = torch.zeros((height, width), device=images.device, dtype=torch.bool)
        for y0, x0 in candidate_yx[selected].tolist():
            disk = ((yy.float() - y0) ** 2 + (xx.float() - x0) ** 2) <= float(spot_radius_px ** 2)
            mask |= disk
        masks.append((mask & valid).float())
    return torch.stack(masks, dim=0).unsqueeze(1)


def make_mask(
    batch_size: int,
    height: int,
    width: int,
    mask_type: str,
    mask_ratio: float,
    patch_size: int,
    device=None,
    images: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
    spot_radius: float = 0.05,
    center_exclude_radius: float = 0.06,
    spot_rank: int = 1,
    peak_window: int = 11,
    spot_selection: str = "random",
    min_spots: int = 1,
    max_spots: int = 4,
    min_peak_fraction: float = 0.05,
) -> torch.Tensor:
    mask_type = mask_type.lower()
    if mask_type == "patch":
        return random_patch_mask(batch_size, height, width, patch_size, mask_ratio, device=device)
    if mask_type in {"wedge", "sector", "angular"}:
        return wedge_mask(batch_size, height, width, mask_ratio, device=device)
    if mask_type in {"central", "disk", "center"}:
        return central_disk_mask(batch_size, height, width, mask_ratio, device=device)
    if mask_type in {"spot", "diffraction_spot", "bragg", "peak"}:
        if images is None:
            raise ValueError("mask_type='spot' requires images to locate diffraction spots")
        return spot_mask(
            images.to(device) if device is not None else images,
            spot_radius=spot_radius,
            center_exclude_radius=center_exclude_radius,
            spot_rank=spot_rank,
            peak_window=peak_window,
            valid_mask=valid_mask,
            spot_selection=spot_selection,
            min_spots=min_spots,
            max_spots=max_spots,
            min_peak_fraction=min_peak_fraction,
        )
    raise ValueError(f"Unknown mask type '{mask_type}'. Expected patch, wedge, central, or spot.")
