import cv2
import numpy as np
import torch.nn.functional as F
import math

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn
from torchvision.transforms import v2


# THE MOST COMPLEX DIFFRACTION TRANSFORM IN THE WORLD
class DiffractionSimTransform(nn.Module):
    def __init__(self, min_disk_contrast=0.01, disk_cutoff_count=64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fwhm_estimator = DiffractionFWHMEstimator()
        self.over_exposure_transform = OverExposureTransform(radius_multiplier=2)
        self.gamma_transform = v2.Compose([
            RandomGammaTransform(gamma_range=(0.5, 1)),
        ])
        self.excitation_manifold_transform = ExcitationManifoldTransform(
            intensity_range=(0.5, 1.0),
            curvature_range=(0.0, 2.0),
            saddle_range=(-1.0, 1.0),
            strain_prob=0.8,
            strain_strength_range=(0.0, 1.0),
            width_range=(0.1, 1.0),
            fringes_range=(1.0, 3.0),
            noise_grid_size=16,
            base_resolution=256
        )
        self.diffraction_scattering_transform = DiffractionScatteringTransform(
            intensity_range=(0.5, 0.75),
            r_fast=(0.1, 2.0),
            r_slow=(0.01, 0.1),
            r_alpha=(0.5, 0.95)
        )
        self.diffraction_amorphous_transform = DiffractionAmorphousTransform(
            intensity_range=(0.01, 0.2),
            halo_count_range=(0, 4),
            probe_scale_range=(2.0, 4.0),
            sigma_scale_range=(0.5, 1)
        )
        self.random_rotation_image_disk_transform = RandomRotationImageDiskTransform()
        self.random_translate_transform = RandomTranslateTransform(random_translation_range=16)
        self.photometric_transformation = v2.Compose([
            RandomGaussianBlur(kernel_size=7, sigma_range=(0.1, 2)),
            v2.CenterCrop((256, 256)),
            SumNormalizationTransform(),
            DiffractionNoiseTransform(dose_range=(1e4, 1e8), sample_log=True),
        ])
        self.min_disk_contrast = min_disk_contrast
        self.disk_cutoff_count = disk_cutoff_count

    def forward(self, image_batch, disk_batch):
        B, C, H, W = image_batch.shape
        device = image_batch.device

        fwhm_batch = self.fwhm_estimator(image_batch) # estimate FWHM
        image_batch = self.over_exposure_transform(image_batch, fwhm_batch) # do over-exposure
        image_batch = self.gamma_transform(image_batch) # do gamma

        # now we sample the disk intensities for later filtering
        ix = disk_batch[..., 0].long().clamp(0, W - 1)
        iy = disk_batch[..., 1].long().clamp(0, H - 1)
        b_idx = torch.arange(B, device=device).view(B, 1).expand(-1, disk_batch.size(1))
        disk_intensity_batch = image_batch[b_idx, 0, iy, ix]
        disk_intensity_batch = disk_intensity_batch * (disk_batch[..., 0] != -1).float()

        # scattering + amorphous part
        image_batch, mask_batch = self.excitation_manifold_transform(image_batch)
        scattering_batch = self.diffraction_scattering_transform(image_batch, None, fwhm_batch)
        amorphous_batch = self.diffraction_amorphous_transform(image_batch, fwhm_batch)
        amorphous_batch = amorphous_batch * mask_batch # note, the amorphous batch is also excitation modulated to give variety

        scattering_img = scattering_batch[0, :, :].squeeze().cpu().numpy()
        scattering_img = (scattering_img - np.min(scattering_img)) / (np.max(scattering_img) - np.min(scattering_img))
        plt.imshow(scattering_img)
        plt.show()
        cv2.imwrite("center_scattering.png", scattering_img * 255)

        amorphous_img = amorphous_batch[0, :, :].squeeze().cpu().numpy()
        amorphous_img = (amorphous_img - np.min(amorphous_img)) / (np.max(amorphous_img) - np.min(amorphous_img))
        plt.imshow(amorphous_img)
        plt.show()
        cv2.imwrite("amorphous_scattering.png", amorphous_img * 255)

        image_img = image_batch[0, :, :].squeeze().cpu().numpy()
        image_img = (image_img - np.min(image_img)) / (np.max(image_img) - np.min(image_img))
        plt.imshow(image_img)
        plt.show()
        cv2.imwrite("crystal_scattering.png", image_img * 255)

        input("")

        background_batch = scattering_batch + amorphous_batch
        image_batch = torch.max(image_batch, background_batch)

        # now we sample the background intensities for later filtering
        background_intensity_batch = background_batch[b_idx, 0, iy, ix]
        background_intensity_batch = background_intensity_batch * (disk_batch[..., 0] != -1).float()
        intensity_mask = disk_intensity_batch >= background_intensity_batch + self.min_disk_contrast

        # now, acquisition transform!
        image_batch, disk_batch = self.random_rotation_image_disk_transform(image_batch, disk_batch)
        image_batch, centroid_batch = self.random_translate_transform(image_batch) # note, ij convention
        image_batch = self.photometric_transformation(image_batch)

        # bias disk coordinate
        new_H, new_W = image_batch.shape[2], image_batch.shape[3]
        old_center = torch.tensor([W - 1, H - 1], device=device, dtype=torch.float32) / 2.0
        old_center = old_center.view(1, 1, 2)
        size_diff = torch.tensor([W - new_W, H - new_H], device=device, dtype=torch.float32) / 2.0
        size_diff = size_diff.view(1, 1, 2)
        new_center_xy = torch.flip(centroid_batch, dims=[-1]).unsqueeze(1) # convert to xy convention
        shift_vector = new_center_xy - old_center - size_diff
        transformed_coords = disk_batch + shift_vector

        # reduce invalid entry
        # 1. Determine Validity
        valid_mask_orig = disk_batch[:, :, 0] != -1
        x_coords = transformed_coords[:, :, 0]
        y_coords = transformed_coords[:, :, 1]
        in_bounds_mask = ((x_coords >= 0) & (x_coords < new_W) & (y_coords >= 0) & (y_coords < new_H))
        final_valid_mask = valid_mask_orig & in_bounds_mask & intensity_mask
        sorting_scores = disk_intensity_batch.clone()
        sorting_scores[~final_valid_mask] = -float('inf')
        num_to_select = min(disk_batch.shape[1], self.disk_cutoff_count)
        top_scores, top_indices = torch.topk(sorting_scores, k=num_to_select, dim=1)
        normalized_x = x_coords / new_W
        normalized_y = y_coords / new_H
        normalized_coords = torch.stack([normalized_x, normalized_y], dim=-1)  # (B, N, 2)
        indices_expanded = top_indices.unsqueeze(-1).expand(-1, -1, 2)
        gathered_coords = torch.gather(normalized_coords, 1, indices_expanded)
        gathered_intensities = torch.gather(disk_intensity_batch, 1, top_indices)
        gathered_bg_intensities = torch.gather(background_intensity_batch, 1, top_indices)
        output_disk_batch = torch.full((B, self.disk_cutoff_count, 2), -1.0, device=disk_batch.device, dtype=disk_batch.dtype)
        output_intensity_batch = torch.full((B, self.disk_cutoff_count), -1.0, device=disk_batch.device, dtype=disk_batch.dtype)
        output_bg_batch = torch.full((B, self.disk_cutoff_count), -1.0, device=disk_batch.device, dtype=disk_batch.dtype)
        output_disk_batch[:, :num_to_select, :] = gathered_coords
        output_intensity_batch[:, :num_to_select] = gathered_intensities
        output_bg_batch[:, :num_to_select] = gathered_bg_intensities

        invalid_in_selection = (top_scores == -float('inf'))
        output_disk_batch[:, :num_to_select, :][invalid_in_selection] = -1.0
        output_intensity_batch[:, :num_to_select][invalid_in_selection] = -1.0
        output_bg_batch[:, :num_to_select][invalid_in_selection] = -1.0

        center_beam_xy = new_center_xy - size_diff
        center_beam_normalized = center_beam_xy.squeeze(1) / torch.tensor([new_W, new_H], device=device, dtype=torch.float32)

        # for i in range(B):
        #     image_preview = image_batch[i, 0, :, :].cpu().squeeze().numpy()
        #     mask_preview = mask_batch[i, 0, :, :].cpu().squeeze().numpy()
        #     scattering_preview = scattering_batch[i, 0, :, :].cpu().squeeze().numpy()
        #     amorphous_preview = amorphous_batch[i, 0, :, :].cpu().squeeze().numpy()
        #     disk_preview = output_disk_batch[i, :, :].cpu().squeeze().numpy()
        #     disk_intensity_preview = output_intensity_batch[i, :].cpu().squeeze().numpy()
        #     center_beam_preview = center_beam_normalized[i, :].cpu().squeeze().numpy()
        #     fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        #     axs[0, 0].imshow(mask_preview, cmap="gray")
        #     axs[0, 0].set_title("Excitation Manifold")
        #     axs[0, 0].axis("off")
        #     axs[0, 1].imshow(scattering_preview, cmap="gray")
        #     axs[0, 1].set_title("Scattering Background")
        #     axs[0, 1].axis("off")
        #     axs[1, 0].imshow(amorphous_preview, cmap="gray")
        #     axs[1, 0].set_title("Amorphous Halo")
        #     axs[1, 0].axis("off")
        #     axs[1, 1].imshow(image_preview, cmap="gray")
        #     valid_mask = disk_intensity_preview != -1
        #     valid_x = disk_preview[valid_mask, 0] * new_W
        #     valid_y = disk_preview[valid_mask, 1] * new_H
        #     valid_int = disk_intensity_preview[valid_mask] * 100
        #     axs[1, 1].scatter(valid_x, valid_y, s=valid_int, c="green", marker=".")
        #     center_beam_x = center_beam_preview[0] * new_W
        #     center_beam_y = center_beam_preview[1] * new_H
        #     axs[1, 1].scatter(center_beam_x, center_beam_y, s=400, c="red", marker="+")
        #     axs[1, 1].set_title("Final Output")
        #     axs[1, 1].axis("off")
        #     plt.tight_layout()
        #     plt.show()
        #     input("Press Enter to view next sample...")

        return image_batch, output_disk_batch, center_beam_normalized


class DiffractionExpTransform(nn.Module):
    def __init__(self, min_disk_contrast=0.01, disk_cutoff_count=64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fwhm_estimator = DiffractionFWHMEstimator()
        self.over_exposure_transform = OverExposureTransform(radius_multiplier=2)
        self.diffraction_scattering_transform = DiffractionScatteringTransform(
            intensity_range=(0.5, 0.75),
            r_fast=(0.1, 2.0),
            r_slow=(0.01, 0.1),
            r_alpha=(0.5, 0.95)
        )
        self.random_rotation_image_disk_transform = RandomRotationImageDiskTransform()
        self.random_translate_transform = RandomTranslateTransform(random_translation_range=16)
        self.photometric_transformation = v2.Compose([
            RandomGaussianBlur(kernel_size=7, sigma_range=(0.1, 2)),
            v2.CenterCrop((256, 256)),
            SumNormalizationTransform(),
            DiffractionNoiseTransform(dose_range=(1e4, 1e8), sample_log=True),
        ])
        self.min_disk_contrast = min_disk_contrast
        self.disk_cutoff_count = disk_cutoff_count

    def forward(self, image_batch, disk_batch):
        B, C, H, W = image_batch.shape
        device = image_batch.device

        fwhm_batch = self.fwhm_estimator(image_batch) # estimate FWHM
        image_batch = self.over_exposure_transform(image_batch, fwhm_batch) # do over-exposure

        # now we sample the disk intensities for later filtering
        ix = disk_batch[..., 0].long().clamp(0, W - 1)
        iy = disk_batch[..., 1].long().clamp(0, H - 1)
        b_idx = torch.arange(B, device=device).view(B, 1).expand(-1, disk_batch.size(1))
        disk_intensity_batch = image_batch[b_idx, 0, iy, ix]
        disk_intensity_batch = disk_intensity_batch * (disk_batch[..., 0] != -1).float()

        # scattering + amorphous part
        scattering_batch = self.diffraction_scattering_transform(image_batch, None, fwhm_batch)
        background_batch = scattering_batch
        image_batch = torch.max(image_batch, background_batch)

        # now we sample the background intensities for later filtering
        background_intensity_batch = background_batch[b_idx, 0, iy, ix]
        background_intensity_batch = background_intensity_batch * (disk_batch[..., 0] != -1).float()
        intensity_mask = disk_intensity_batch >= background_intensity_batch + self.min_disk_contrast

        # now, acquisition transform!
        image_batch, disk_batch = self.random_rotation_image_disk_transform(image_batch, disk_batch)
        image_batch, centroid_batch = self.random_translate_transform(image_batch) # note, ij convention
        image_batch = self.photometric_transformation(image_batch)

        # bias disk coordinate
        new_H, new_W = image_batch.shape[2], image_batch.shape[3]
        old_center = torch.tensor([W - 1, H - 1], device=device, dtype=torch.float32) / 2.0
        old_center = old_center.view(1, 1, 2)
        size_diff = torch.tensor([W - new_W, H - new_H], device=device, dtype=torch.float32) / 2.0
        size_diff = size_diff.view(1, 1, 2)
        new_center_xy = torch.flip(centroid_batch, dims=[-1]).unsqueeze(1) # convert to xy convention
        shift_vector = new_center_xy - old_center - size_diff
        transformed_coords = disk_batch + shift_vector

        # reduce invalid entry
        # 1. Determine Validity
        valid_mask_orig = disk_batch[:, :, 0] != -1
        x_coords = transformed_coords[:, :, 0]
        y_coords = transformed_coords[:, :, 1]
        in_bounds_mask = ((x_coords >= 0) & (x_coords < new_W) & (y_coords >= 0) & (y_coords < new_H))
        final_valid_mask = valid_mask_orig & in_bounds_mask & intensity_mask
        sorting_scores = disk_intensity_batch.clone()
        sorting_scores[~final_valid_mask] = -float('inf')
        num_to_select = min(disk_batch.shape[1], self.disk_cutoff_count)
        top_scores, top_indices = torch.topk(sorting_scores, k=num_to_select, dim=1)
        normalized_x = x_coords / new_W
        normalized_y = y_coords / new_H
        normalized_coords = torch.stack([normalized_x, normalized_y], dim=-1)  # (B, N, 2)
        indices_expanded = top_indices.unsqueeze(-1).expand(-1, -1, 2)
        gathered_coords = torch.gather(normalized_coords, 1, indices_expanded)
        gathered_intensities = torch.gather(disk_intensity_batch, 1, top_indices)
        gathered_bg_intensities = torch.gather(background_intensity_batch, 1, top_indices)
        output_disk_batch = torch.full((B, self.disk_cutoff_count, 2), -1.0, device=disk_batch.device, dtype=disk_batch.dtype)
        output_intensity_batch = torch.full((B, self.disk_cutoff_count), -1.0, device=disk_batch.device, dtype=disk_batch.dtype)
        output_bg_batch = torch.full((B, self.disk_cutoff_count), -1.0, device=disk_batch.device, dtype=disk_batch.dtype)
        output_disk_batch[:, :num_to_select, :] = gathered_coords
        output_intensity_batch[:, :num_to_select] = gathered_intensities
        output_bg_batch[:, :num_to_select] = gathered_bg_intensities

        invalid_in_selection = (top_scores == -float('inf'))
        output_disk_batch[:, :num_to_select, :][invalid_in_selection] = -1.0
        output_intensity_batch[:, :num_to_select][invalid_in_selection] = -1.0
        output_bg_batch[:, :num_to_select][invalid_in_selection] = -1.0

        center_beam_xy = new_center_xy - size_diff
        center_beam_normalized = center_beam_xy.squeeze(1) / torch.tensor([new_W, new_H], device=device, dtype=torch.float32)

        # for i in range(B):
        #     image_preview = image_batch[i, 0, :, :].cpu().squeeze().numpy()
        #     scattering_preview = scattering_batch[i, 0, :, :].cpu().squeeze().numpy()
        #     disk_preview = output_disk_batch[i, :, :].cpu().squeeze().numpy()
        #     disk_intensity_preview = output_intensity_batch[i, :].cpu().squeeze().numpy()
        #     center_beam_preview = center_beam_normalized[i, :].cpu().squeeze().numpy()
        #     fig, axs = plt.subplots(1, 2, figsize=(10, 10))
        #     axs[0].imshow(scattering_preview, cmap="gray")
        #     axs[0].set_title("Scattering Background")
        #     axs[0].axis("off")
        #     axs[1].imshow(image_preview, cmap="gray")
        #     valid_mask = disk_intensity_preview != -1
        #     valid_x = disk_preview[valid_mask, 0] * new_W
        #     valid_y = disk_preview[valid_mask, 1] * new_H
        #     valid_int = disk_intensity_preview[valid_mask] * 100
        #     axs[1].scatter(valid_x, valid_y, s=valid_int, c="green", marker=".")
        #     center_beam_x = center_beam_preview[0] * new_W
        #     center_beam_y = center_beam_preview[1] * new_H
        #     axs[1].scatter(center_beam_x, center_beam_y, s=400, c="red", marker="+")
        #     axs[1].set_title("Final Output")
        #     axs[1].axis("off")
        #     plt.tight_layout()
        #     plt.show()
        #     input("Press Enter to view next sample...")

        return image_batch, output_disk_batch, center_beam_normalized


class DiffractionFWHMEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self._grid_cache = {}  # Cache for (H, W, device) -> (grid_i, grid_j, R_base)

    def _get_cached_grids(self, H, W, device):
        cache_key = (H, W, str(device))
        if cache_key not in self._grid_cache:
            ci_float = (H - 1) / 2.0
            cj_float = (W - 1) / 2.0
            i_axis = torch.arange(H, device=device).float()
            j_axis = torch.arange(W, device=device).float()
            grid_i, grid_j = torch.meshgrid(i_axis, j_axis, indexing='ij')
            grid_i = grid_i.unsqueeze(0).unsqueeze(0)
            grid_j = grid_j.unsqueeze(0).unsqueeze(0)
            R_base = torch.sqrt((grid_i - ci_float) ** 2 + (grid_j - cj_float) ** 2)
            self._grid_cache[cache_key] = (grid_i, grid_j, R_base)
        return self._grid_cache[cache_key]

    def get_batch_radial_profile(self, image_batch, R, max_r):
        B, C, H, W = image_batch.shape
        device = image_batch.device
        if C > 1:
            img_spatial = image_batch.mean(dim=1)
        else:
            img_spatial = image_batch.squeeze(1)
        pixel_vals = img_spatial.reshape(-1)
        r_bins = R.squeeze(1).long()
        r_bins = torch.clamp(r_bins, 0, max_r - 1)
        batch_ids = torch.arange(B, device=device).view(B, 1, 1).expand(B, H, W)
        global_bin_ids = (batch_ids * max_r + r_bins).reshape(-1)
        total_bins = B * max_r
        bin_sums = torch.zeros(total_bins, device=device)
        bin_sums.index_add_(0, global_bin_ids, pixel_vals)
        bin_counts = torch.zeros(total_bins, device=device)
        bin_counts.index_add_(0, global_bin_ids, torch.ones_like(pixel_vals))
        bin_counts = torch.clamp(bin_counts, min=1.0)
        profiles_flat = bin_sums / bin_counts
        profiles = profiles_flat.view(B, max_r)
        return profiles

    def calculate_fwhm_radius(self, profiles, reference_intensities):
        B, max_r = profiles.shape
        targets = (reference_intensities / 2.0).view(B, 1)
        below_target = profiles < targets
        idx_drop = torch.argmax(below_target.float(), dim=1)
        valid_drop = below_target.gather(1, idx_drop.unsqueeze(1)).squeeze(1)
        idx_r2 = torch.clamp(idx_drop, 1, max_r - 1)
        idx_r1 = idx_r2 - 1
        v2 = profiles.gather(1, idx_r2.unsqueeze(1)).squeeze(1)
        v1 = profiles.gather(1, idx_r1.unsqueeze(1)).squeeze(1)
        r1 = idx_r1.float()
        slope = v2 - v1
        slope = torch.where(slope == 0, torch.tensor(1e-6, device=profiles.device), slope)
        fraction = (targets.squeeze(1) - v1) / slope
        measured_r = r1 + fraction
        final_r = torch.where(valid_drop, measured_r, torch.tensor(5.0, device=profiles.device))
        return final_r.view(B, 1, 1, 1)

    def forward(self, image_batch):
        if image_batch.dim() == 3:
            image_batch = image_batch.unsqueeze(1)
        B, C, H, W = image_batch.shape
        device = image_batch.device
        ci_int = H // 2
        cj_int = W // 2
        grid_i, grid_j, R = self._get_cached_grids(H, W, device)
        if C > 1:
            v_centroid = image_batch[:, :, ci_int, cj_int].mean(dim=1)
        else:
            v_centroid = image_batch[:, 0, ci_int, cj_int]
        max_scan_r = min(H, W) // 2
        profiles = self.get_batch_radial_profile(image_batch, R, max_scan_r)
        fwhm_batch = self.calculate_fwhm_radius(profiles, v_centroid)
        return fwhm_batch


class DiffractionAmorphousTransform(nn.Module):
    def __init__(self, intensity_range=(0.05, 0.2), halo_count_range=(0, 3), probe_scale_range=(3.0, 10.0), sigma_scale_range=(0.5, 2.0)):
        super().__init__()
        self.intensity_range = intensity_range
        self.halo_count_range = halo_count_range
        self.probe_scale_range = probe_scale_range
        self.sigma_scale_range = sigma_scale_range
        self._grid_cache = {}  # Cache for (H, W, device) -> r

    def _get_cached_radial_grid(self, H, W, device):
        cache_key = (H, W, str(device))
        if cache_key not in self._grid_cache:
            ci = (H - 1) / 2.0
            cj = (W - 1) / 2.0
            i_axis = torch.arange(H, device=device).float()
            j_axis = torch.arange(W, device=device).float()
            grid_i, grid_j = torch.meshgrid(i_axis, j_axis, indexing='ij')
            y = grid_i - ci
            x = grid_j - cj
            r = torch.sqrt(x ** 2 + y ** 2).unsqueeze(0).unsqueeze(0)
            self._grid_cache[cache_key] = r
        return self._grid_cache[cache_key]

    def forward(self, image_batch, fwhm_batch):
        B, C, H, W = image_batch.shape
        device = image_batch.device

        r = self._get_cached_radial_grid(H, W, device)
        num_halos_per_image = torch.randint(
            self.halo_count_range[0],
            self.halo_count_range[1] + 1,
            (B,),
            device=device
        )
        max_halos = num_halos_per_image.max().item()
        amorphous_batch = torch.zeros((B, 1, H, W), device=device)
        if fwhm_batch.dim() == 1:
            fwhm_batch = fwhm_batch.view(B, 1, 1, 1)
        for k in range(max_halos):
            active_mask = (num_halos_per_image > k).float().view(B, 1, 1, 1)
            scale_r = torch.rand(B, 1, 1, 1, device=device) * (self.probe_scale_range[1] - self.probe_scale_range[0]) + self.probe_scale_range[0]
            center_r = fwhm_batch * scale_r
            scale_sigma = torch.rand(B, 1, 1, 1, device=device) * (self.sigma_scale_range[1] - self.sigma_scale_range[0]) + self.sigma_scale_range[0]
            sigma = fwhm_batch * scale_sigma
            intensity = torch.rand(B, 1, 1, 1, device=device) * (self.intensity_range[1] - self.intensity_range[0]) + self.intensity_range[0]
            radial_profile = torch.exp(-0.5 * ((r - center_r) / sigma) ** 2)
            amorphous_batch = torch.max(amorphous_batch, (radial_profile * intensity) * active_mask)
        if C > 1:
            amorphous_batch = amorphous_batch.expand(-1, C, -1, -1)
        return amorphous_batch


class DiffractionScatteringTransform(nn.Module):
    def __init__(self, intensity_range=(0.5, 1.0), r_fast=(0.1, 2.0), r_slow=(0.01, 0.1), r_alpha=(0.5, 0.95)):
        super().__init__()
        self.intensity_range = intensity_range
        self.r_fast = r_fast
        self.r_slow = r_slow
        self.r_alpha = r_alpha
        self._grid_cache = {}  # Cache for (H, W, device) -> (grid_i, grid_j)

    def _get_cached_grids(self, H, W, device):
        cache_key = (H, W, str(device))
        if cache_key not in self._grid_cache:
            i_axis = torch.arange(H, device=device).float()
            j_axis = torch.arange(W, device=device).float()
            grid_i, grid_j = torch.meshgrid(i_axis, j_axis, indexing='ij')
            grid_i = grid_i.unsqueeze(0).unsqueeze(0)
            grid_j = grid_j.unsqueeze(0).unsqueeze(0)
            self._grid_cache[cache_key] = (grid_i, grid_j)
        return self._grid_cache[cache_key]

    def generate_physics_params(self, batch_size, device):
        def sample_uniform(low, high):
            return (high - low) * torch.rand(batch_size, 1, 1, 1, device=device) + low
        alpha = sample_uniform(*self.r_alpha)
        k1 = sample_uniform(*self.r_fast)
        k2 = sample_uniform(*self.r_slow)
        k_fast = torch.max(k1, k2)
        k_slow = torch.min(k1, k2)
        intensity = sample_uniform(*self.intensity_range)
        return alpha, k_fast, k_slow, intensity

    def forward(self, image_batch, centroid_batch, fwhm_batch):
        if image_batch.dim() == 3:
            image_batch = image_batch.unsqueeze(1)
        B, C, H, W = image_batch.shape
        device = image_batch.device

        if centroid_batch is None:
            center_i = (H - 1.0) / 2.0
            center_j = (W - 1.0) / 2.0
            centroid_batch = torch.tensor([[center_i, center_j]], device=device).repeat(B, 1)

        ci = centroid_batch[:, 0].view(B, 1, 1, 1)
        cj = centroid_batch[:, 1].view(B, 1, 1, 1)
        grid_i, grid_j = self._get_cached_grids(H, W, device)
        R = torch.sqrt((grid_i - ci) ** 2 + (grid_j - cj) ** 2)
        samp_i = torch.clamp(centroid_batch[:, 0].long(), 0, H - 1)
        samp_j = torch.clamp(centroid_batch[:, 1].long(), 0, W - 1)
        batch_indices = torch.arange(B, device=device)
        v_centroid = image_batch[batch_indices, 0, samp_i, samp_j]
        alpha, k_fast, k_slow, intensity = self.generate_physics_params(B, device)
        I_plat = v_centroid.view(B, 1, 1, 1) * intensity
        mask_core = (R <= fwhm_batch)
        mask_tail = ~mask_core
        dr = R - fwhm_batch
        dr = torch.relu(dr)
        term_fast = alpha * torch.exp(-k_fast * dr)
        term_slow = (1.0 - alpha) * torch.exp(-k_slow * dr)
        decay_vals = I_plat * (term_fast + term_slow)
        scattering_batch = torch.zeros_like(image_batch)
        scattering_batch = torch.where(mask_core, I_plat, scattering_batch)
        scattering_batch = torch.where(mask_tail, decay_vals, scattering_batch)
        return scattering_batch


class ExcitationManifoldTransform(nn.Module):
    # note, these params are set using base resolution 256.
    def __init__(self,
                 intensity_range=(0.5, 1.0),
                 curvature_range=(0.0, 2.0),
                 saddle_range=(-1.0, 1.0),
                 strain_prob=0.8,
                 strain_strength_range=(0.0, 1.0),
                 width_range=(0.1, 1.0),
                 fringes_range=(1.0, 3.0),
                 noise_grid_size=16,
                 base_resolution=256):
        super().__init__()
        self.intensity_range = intensity_range
        self.curv_range = curvature_range
        self.saddle_range = saddle_range
        self.strain_prob = strain_prob
        self.strain_str_range = strain_strength_range
        self.width_range = width_range
        self.fringes_range = fringes_range
        self.noise_grid_size = noise_grid_size
        self.base_resolution = float(base_resolution)
        self._grid_cache = {}  # Cache for (H, W, device) -> (i_lin, j_lin, grid_i, grid_j)

    def _get_cached_grids(self, H, W, device):
        cache_key = (H, W, str(device))
        if cache_key not in self._grid_cache:
            scale_h = H / self.base_resolution
            scale_w = W / self.base_resolution
            i_lin = torch.linspace(-scale_h, scale_h, steps=H, device=device)
            j_lin = torch.linspace(-scale_w, scale_w, steps=W, device=device)
            grid_i, grid_j = torch.meshgrid(i_lin, j_lin, indexing='ij')
            self._grid_cache[cache_key] = (i_lin, j_lin, grid_i, grid_j)
        return self._grid_cache[cache_key]

    def get_random_params(self, B, device):
        def rand_range(r):
            return (r[1] - r[0]) * torch.rand(B, 1, 1, device=device) + r[0]
        p_intensity = rand_range(self.intensity_range)
        p_curv = rand_range(self.curv_range)
        p_saddle = rand_range(self.saddle_range)
        p_strain = rand_range(self.strain_str_range)
        p_width = rand_range(self.width_range)
        p_fringes = rand_range(self.fringes_range)
        theta = torch.rand(B, 1, 1, device=device) * math.pi
        p_cubic = torch.randn(B, 1, 1, device=device) * 0.5
        off_x = 0.5 * (torch.rand(B, 1, 1, device=device) - 0.5)
        off_y = 0.5 * (torch.rand(B, 1, 1, device=device) - 0.5)
        phi = torch.rand(B, 1, 1, device=device) * 2 * math.pi
        return p_intensity, p_curv, p_saddle, p_strain, p_width, p_fringes, theta, p_cubic, off_x, off_y, phi

    def forward(self, image_batch):
        if image_batch.dim() == 3:
            image_batch = image_batch.unsqueeze(1)
        B, C, H, W = image_batch.shape
        device = image_batch.device
        scale_h = H / self.base_resolution
        scale_w = W / self.base_resolution
        effective_noise_h = int(self.noise_grid_size * scale_h)
        effective_noise_w = int(self.noise_grid_size * scale_w)
        effective_noise_h = max(1, effective_noise_h)
        effective_noise_w = max(1, effective_noise_w)
        (p_int, p_curv, p_saddle, p_strain_amp, p_width, p_fringes, theta, p_cubic, off_x, off_y, phi) = self.get_random_params(B, device)
        i_lin, j_lin, grid_i, grid_j = self._get_cached_grids(H, W, device)
        Y = grid_i.unsqueeze(0).expand(B, -1, -1) - off_y
        X = grid_j.unsqueeze(0).expand(B, -1, -1) - off_x
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        X_rot = X * cos_t - Y * sin_t
        Y_rot = X * sin_t + Y * cos_t
        y_coeff = -p_saddle
        S_global = p_curv * (X_rot ** 2 + y_coeff * (Y_rot ** 2) + p_cubic * (X_rot ** 3))
        S_mean = S_global.view(B, -1).mean(dim=1).view(B, 1, 1)
        S_global = S_global - S_mean
        noise_small = torch.randn(B, 1, effective_noise_h, effective_noise_w, device=device)
        noise_smooth = F.interpolate(noise_small, size=(H, W), mode='bicubic', align_corners=False).squeeze(1)
        S_total = S_global + (noise_smooth * p_strain_amp * 2.0)
        w = S_total / p_width
        wedge_mag = torch.rand(B, 1, 1, device=device)
        wedge = 1.0 + wedge_mag * (X * torch.cos(phi) + Y * torch.sin(phi))
        envelope = 1.0 / (1.0 + w ** 2)
        fringe_arg = p_fringes * wedge * torch.sqrt(1.0 + w ** 2)
        fringes = torch.sin(fringe_arg) ** 2
        dyn_mask = envelope * fringes
        max_vals = dyn_mask.view(B, -1).max(dim=1).values.view(B, 1, 1)
        max_vals = torch.clamp(max_vals, min=1e-6)
        dyn_mask = dyn_mask / max_vals
        dyn_mask = 1.0 - dyn_mask
        dyn_mask = torch.clamp(dyn_mask, 0.0, 1.0).unsqueeze(1)  # (B, 1, H, W)
        p_int_4d = p_int.view(B, 1, 1, 1)
        effective_mask = dyn_mask * p_int_4d + (1.0 - p_int_4d)
        output = image_batch * effective_mask
        return output, effective_mask


class DiffractionNoiseTransform(torch.nn.Module):
    def __init__(self, dose_range=(1e5, 1e9), sigma_range=(0, 0), bias_range=(0, 0), sample_log=True):
        super().__init__()
        self.dose_range = dose_range
        self.sigma_range = sigma_range
        self.bias_range = bias_range
        self.sample_log = sample_log

    def get_random_dose(self, batch_size, device):
        min_dose, max_dose = self.dose_range
        if self.sample_log:
            log_min = math.log10(min_dose)
            log_max = math.log10(max_dose)
            random_exponents = torch.rand(batch_size, device=device) * (log_max - log_min) + log_min
            doses = 10 ** random_exponents
        else:
            doses = torch.rand(batch_size, device=device) * (max_dose - min_dose) + min_dose
        return doses.view(-1, 1, 1, 1)  # Shape for broadcasting

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        if image_batch.dim() == 3:
            image_batch = image_batch.unsqueeze(1)
        B, C, H, W = image_batch.shape
        device = image_batch.device
        epsilon = 1e-8
        dose = self.get_random_dose(B, device)
        sigma = torch.rand(B, 1, 1, 1, device=device) * (self.sigma_range[1] - self.sigma_range[0]) + self.sigma_range[0]
        bias = torch.rand(B, 1, 1, 1, device=device) * (self.bias_range[1] - self.bias_range[0]) + self.bias_range[0]
        scaled_input = torch.clamp(image_batch, min=0) * dose
        image_batch_noisy = torch.poisson(scaled_input)
        readout_noise = torch.randn_like(image_batch_noisy) * sigma + bias
        image_batch_noisy = image_batch_noisy + readout_noise
        batch_max = image_batch_noisy.view(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        image_batch_final = image_batch_noisy / (batch_max + epsilon)
        return torch.clamp(image_batch_final, 0, 1)


class SumNormalizationTransform(torch.nn.Module):
    # Normalize to probability map
    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        B, C, H, W = image_batch.shape
        image_batch_flat = image_batch.view(B, C, H, W)
        batch_sum = torch.sum(image_batch_flat, dim=(1, 2, 3), keepdim=True)
        normalized_batch = image_batch_flat / (batch_sum + self.epsilon)
        return normalized_batch.view(B, C, H, W)


class MaxNormalizationTransform(torch.nn.Module):
    # Normalize the diffraction pattern
    def __init__(self):
        super().__init__()

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        B, C, H, W = image_batch.shape
        image_batch_flat = image_batch.view(B * C, H, W)
        epsilon = 1e-8
        batch_max = image_batch_flat.view(B, -1).max(dim=1, keepdim=True)[0]
        image_batch_final = image_batch_flat.view(B, -1) / (batch_max + epsilon)
        return torch.clamp(image_batch_final, 0, 1).view(B, C, H, W)


class MinMaxNormalizationTransform(torch.nn.Module):
    # Normalize the diffraction pattern
    def __init__(self):
        super().__init__()

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        B, C, H, W = image_batch.shape
        image_batch_flat = image_batch.view(B * C, H, W)
        epsilon = 1e-8
        batch_min = image_batch_flat.reshape(B, -1).min(dim=1, keepdim=True)[0]
        batch_max = image_batch_flat.reshape(B, -1).max(dim=1, keepdim=True)[0]
        image_batch_final = (image_batch_flat.reshape(B, -1) - batch_min) / (batch_max - batch_min + epsilon)
        return torch.clamp(image_batch_final, 0, 1).view(B, C, H, W)


class OverExposureTransform(nn.Module):
    def __init__(self, radius_multiplier=1.0):
        super().__init__()
        self.radius_multiplier = radius_multiplier

    def forward(self, image_batch: torch.Tensor, fwhm_batch: torch.Tensor) -> torch.Tensor:
        B, C, H, W = image_batch.shape
        device = image_batch.device
        r = fwhm_batch * self.radius_multiplier
        probe_area = torch.pi * (r ** 2)
        total_pixels = float(H * W)
        coverage_ratio = probe_area / total_pixels
        quantile_batch = 1.0 - coverage_ratio
        quantile_batch = torch.clamp(quantile_batch, 0.0, 0.9999)
        a = torch.rand(B, 1, 1, 1, device=device)
        final_quantiles = quantile_batch * a + (1.0 - a) * 1.0
        flat_images = image_batch.reshape(B, -1)
        sorted_vals, _ = torch.sort(flat_images, dim=1)
        idx = (final_quantiles.view(B, 1) * flat_images.shape[1]).long()
        idx = torch.clamp(idx, 0, flat_images.shape[1] - 1)
        thresholds = sorted_vals.gather(1, idx).view(B, 1, 1, 1)
        image_clipped = torch.min(image_batch, thresholds)
        vals_min = flat_images.min(dim=1)[0].view(B, 1, 1, 1)
        denominator = thresholds - vals_min
        denominator = torch.where(denominator < 1e-6, torch.tensor(1.0, device=device), denominator)
        image_normalized = (image_clipped - vals_min) / denominator
        return image_normalized


class RandomTranslateTransform(nn.Module):
    def __init__(self, random_translation_range: float):
        super().__init__()
        if random_translation_range < 0:
            raise ValueError("random_translation_range must be non-negative.")
        self.random_translation_range = random_translation_range

    def forward(self, image_batch: torch.Tensor):
        B, C, H, W = image_batch.shape
        device = image_batch.device
        dtype = image_batch.dtype
        center_i = (H - 1.0) / 2.0
        center_j = (W - 1.0) / 2.0
        if self.random_translation_range == 0:
            centers = torch.tensor([[center_i, center_j]], device=device, dtype=dtype).repeat(B, 1)
            return image_batch, centers
        random_offsets_pixels = (torch.rand(B, 2, device=device) * 2 - 1) * self.random_translation_range
        offset_j = random_offsets_pixels[:, 0]
        offset_i = random_offsets_pixels[:, 1]
        new_ci = center_i + offset_i
        new_cj = center_j + offset_j
        new_centers = torch.stack((new_ci, new_cj), dim=1)
        image_size = torch.tensor([W, H], device=device, dtype=torch.float32)
        normalized_translation = random_offsets_pixels / (image_size / 2.0)
        affine_matrix_batch = torch.zeros(B, 2, 3, device=device, dtype=dtype)
        affine_matrix_batch[:, 0, 0] = 1.0
        affine_matrix_batch[:, 1, 1] = 1.0
        affine_matrix_batch[:, :, 2] = -normalized_translation
        grid_batch = F.affine_grid(affine_matrix_batch, image_batch.shape, align_corners=False)
        transformed_batch = F.grid_sample(
            image_batch,
            grid_batch,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )
        return transformed_batch, new_centers


class RandomGammaTransform(nn.Module):
    def __init__(self, gamma_range=(0.5, 1.0)):
        super().__init__()
        self.gamma_range = gamma_range

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        if self.gamma_range[0] == 1.0 and self.gamma_range[1] == 1.0:
            return image_batch
        B = image_batch.shape[0]
        device = image_batch.device
        low, high = self.gamma_range
        gamma = (high - low) * torch.rand(B, 1, 1, 1, device=device) + low
        return image_batch.clamp(min=1e-8) ** gamma


class RandomGaussianBlur(nn.Module):
    def __init__(self, kernel_size=5, sigma_range=(0.1, 2.0)):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        self.sigma_range = sigma_range

    def get_gaussian_kernel1d(self, kernel_size, sigma):
        k_half = kernel_size // 2
        x = torch.arange(-k_half, k_half + 1, device=sigma.device).float()
        x = x.unsqueeze(0)
        sigma = sigma.unsqueeze(1)
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / kernel.sum(dim=1, keepdim=True)
        return kernel

    def forward(self, image_batch):
        if image_batch.dim() == 3:
            image_batch = image_batch.unsqueeze(1)
        B, C, H, W = image_batch.shape
        device = image_batch.device
        low, high = self.sigma_range
        sigma = torch.rand(B, device=device) * (high - low) + low
        k1d = self.get_gaussian_kernel1d(self.kernel_size, sigma)
        kernel_x = k1d.view(B, 1, 1, self.kernel_size)
        kernel_y = k1d.view(B, 1, self.kernel_size, 1)
        if C > 1:
            kernel_x = kernel_x.repeat_interleave(C, dim=0)
            kernel_y = kernel_y.repeat_interleave(C, dim=0)
        pad = self.kernel_size // 2
        img_reshape = image_batch.view(1, B * C, H, W)
        interim = F.conv2d(img_reshape, kernel_x, padding=(0, pad), groups=B * C)
        final = F.conv2d(interim, kernel_y, padding=(pad, 0), groups=B * C)
        return final.view(B, C, H, W)


class RandomRotation(nn.Module):
    def __init__(self, angle_range=(-30, 30)):
        super().__init__()
        self.angle_range = angle_range
    def forward(self, image_batch):
        if self.angle_range[0] == 0 and self.angle_range[1] == 0:
            return image_batch
        B, C, H, W = image_batch.shape
        device = image_batch.device
        min_a, max_a = self.angle_range
        thetas_deg = torch.rand(B, device=device) * (max_a - min_a) + min_a
        thetas = torch.deg2rad(thetas_deg)
        aspect = W / float(H)
        cos_t = torch.cos(thetas)
        sin_t = torch.sin(thetas)
        affine_matrices = torch.zeros(B, 2, 3, device=device)
        affine_matrices[:, 0, 0] = cos_t
        affine_matrices[:, 0, 1] = -sin_t / aspect
        affine_matrices[:, 1, 0] = sin_t * aspect
        affine_matrices[:, 1, 1] = cos_t
        grid = F.affine_grid(affine_matrices, image_batch.shape, align_corners=False)
        output = F.grid_sample(image_batch, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        return output


class RandomRotationImageDiskTransform(nn.Module):
    def __init__(self, angle_range=(-180, 180)):
        super().__init__()
        self.angle_range = angle_range

    def forward(self, image_batch, disk_batch):
        if self.angle_range[0] == 0 and self.angle_range[1] == 0:
            return image_batch, disk_batch
        B, C, H, W = image_batch.shape
        device = image_batch.device
        dtype = image_batch.dtype
        min_a, max_a = self.angle_range
        angles_deg = torch.rand(B, device=device, dtype=dtype) * (max_a - min_a) + min_a
        angles_rad = torch.deg2rad(angles_deg)
        cos_a = torch.cos(angles_rad)
        sin_a = torch.sin(angles_rad)
        aspect_ratio = W / float(H)
        rot_mat = torch.zeros(B, 2, 3, device=device, dtype=dtype)
        rot_mat[:, 0, 0] = cos_a
        rot_mat[:, 0, 1] = -sin_a / aspect_ratio
        rot_mat[:, 1, 0] = sin_a * aspect_ratio
        rot_mat[:, 1, 1] = cos_a
        grid = F.affine_grid(rot_mat, image_batch.shape, align_corners=False)
        rotated_image_batch = F.grid_sample(
            image_batch,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )
        cx = (W - 1) / 2.0
        cy = (H - 1) / 2.0
        x = disk_batch[..., 0] - cx
        y = disk_batch[..., 1] - cy
        cos_a_v = cos_a.view(B, 1)
        sin_a_v = sin_a.view(B, 1)
        x_new = x * cos_a_v + y * sin_a_v
        y_new = -x * sin_a_v + y * cos_a_v
        x_final = x_new + cx
        y_final = y_new + cy
        valid_mask = (disk_batch[..., 0] != -1)
        rotated_disk_batch = torch.full_like(disk_batch, -1.0)
        rotated_coords = torch.stack([x_final, y_final], dim=-1)
        rotated_disk_batch = torch.where(
            valid_mask.unsqueeze(-1),
            rotated_coords,
            rotated_disk_batch
        )
        return rotated_image_batch, rotated_disk_batch