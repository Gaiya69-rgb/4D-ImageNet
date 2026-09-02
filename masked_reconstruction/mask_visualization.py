import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from dataset import DiffractionPatternDataset, build_sample_index
from masks import make_mask


# One-key F5 defaults. Edit this block, then open this file in VS Code and press F5.
SCRIPT_DEFAULTS = {
    "data_root": Path(r"G:\4D-ImageNet\sim\CuO"),
    "metadata": None,
    "file_index": None,
    "output_dir": Path(r"G:\4D-ImageNet\encoder\outputs\mask_visualization"),
    "image_size": None,  # None keeps the native diffraction-pattern size.
    "max_files": 1,
    "max_samples": 256,
    "num_examples": 6,
    "seed": 42,
    "selection_mode": "rich",  # rich, sequential, random
    "display_percentile": 99.5,  # lower values reveal weaker diffraction spots near the center beam.
    "mask_fill_value": 0.35,  # gray level in display coordinates; 0.35 is medium gray.
    "show_window": True,

    # Spot-targeted mask settings. Values < 1 are interpreted as a fraction of min(H, W).
    "spot_radius": 0.035,
    "center_exclude_radius": 0.06,
    "spot_rank": 1,  # 1 = brightest non-central spot, 2 = second brightest, etc.
    "peak_window": 11,
    "spot_selection": "random",  # random, nearest_bright, or brightest
    "min_spots": 1,
    "max_spots": 4,
    "min_peak_fraction": 0.12,
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Visualize spot-masked diffraction-pattern inputs")
    parser.add_argument("--data-root", default=SCRIPT_DEFAULTS["data_root"], type=Path)
    parser.add_argument("--metadata", default=SCRIPT_DEFAULTS["metadata"], type=Path)
    parser.add_argument("--file-index", default=SCRIPT_DEFAULTS["file_index"], type=Path)
    parser.add_argument("--output-dir", default=SCRIPT_DEFAULTS["output_dir"], type=Path)
    parser.add_argument("--image-size", default=SCRIPT_DEFAULTS["image_size"], type=int)
    parser.add_argument("--max-files", default=SCRIPT_DEFAULTS["max_files"], type=int)
    parser.add_argument("--max-samples", default=SCRIPT_DEFAULTS["max_samples"], type=int)
    parser.add_argument("--num-examples", default=SCRIPT_DEFAULTS["num_examples"], type=int)
    parser.add_argument("--seed", default=SCRIPT_DEFAULTS["seed"], type=int)
    parser.add_argument("--selection-mode", default=SCRIPT_DEFAULTS["selection_mode"], choices=["rich", "sequential", "random"])
    parser.add_argument("--display-percentile", default=SCRIPT_DEFAULTS["display_percentile"], type=float)
    parser.add_argument("--mask-fill-value", default=SCRIPT_DEFAULTS["mask_fill_value"], type=float)
    parser.add_argument("--show-window", action=argparse.BooleanOptionalAction, default=SCRIPT_DEFAULTS["show_window"])
    parser.add_argument("--spot-radius", default=SCRIPT_DEFAULTS["spot_radius"], type=float)
    parser.add_argument("--center-exclude-radius", default=SCRIPT_DEFAULTS["center_exclude_radius"], type=float)
    parser.add_argument("--spot-rank", default=SCRIPT_DEFAULTS["spot_rank"], type=int)
    parser.add_argument("--peak-window", default=SCRIPT_DEFAULTS["peak_window"], type=int)
    parser.add_argument("--spot-selection", default=SCRIPT_DEFAULTS["spot_selection"], choices=["random", "nearest_bright", "brightest"])
    parser.add_argument("--min-spots", default=SCRIPT_DEFAULTS["min_spots"], type=int)
    parser.add_argument("--max-spots", default=SCRIPT_DEFAULTS["max_spots"], type=int)
    parser.add_argument("--min-peak-fraction", default=SCRIPT_DEFAULTS["min_peak_fraction"], type=float)
    return parser



def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().squeeze().numpy()


def diffraction_richness(image: torch.Tensor, center_exclude_radius: float) -> float:
    arr = to_numpy(image[0])
    height, width = arr.shape
    yy, xx = np.ogrid[:height, :width]
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    radius = center_exclude_radius * min(height, width) if center_exclude_radius < 1 else center_exclude_radius
    outside_center = (yy - cy) ** 2 + (xx - cx) ** 2 > radius ** 2
    values = arr[outside_center]
    if values.size == 0:
        return 0.0
    high = values > np.quantile(values, 0.995)
    return float(values[high].sum() + 0.1 * high.sum())


def sample_indices(dataset, num_examples: int, selection_mode: str, seed: int, center_exclude_radius: float):
    num_items = len(dataset)
    indices = list(range(num_items))
    if selection_mode == "random":
        rng = random.Random(seed)
        rng.shuffle(indices)
        return indices[: min(num_examples, num_items)]
    if selection_mode == "sequential":
        return indices[: min(num_examples, num_items)]

    scored = []
    for idx in indices:
        image = dataset[idx]["image"]
        scored.append((diffraction_richness(image, center_exclude_radius), idx))
    scored.sort(reverse=True)
    return [idx for _, idx in scored[: min(num_examples, num_items)]]


def make_visual_mask(image: torch.Tensor, args):
    _, height, width = image.shape
    return make_mask(
        1,
        height,
        width,
        "spot",
        mask_ratio=0.5,
        patch_size=16,
        device=image.device,
        images=image.unsqueeze(0),
        spot_radius=args.spot_radius,
        center_exclude_radius=args.center_exclude_radius,
        spot_rank=args.spot_rank,
        peak_window=args.peak_window,
        spot_selection=args.spot_selection,
        min_spots=args.min_spots,
        max_spots=args.max_spots,
        min_peak_fraction=args.min_peak_fraction,
    )[0]


def mask_center(mask: torch.Tensor):
    coords = torch.nonzero(mask[0] > 0, as_tuple=False)
    if coords.numel() == 0:
        return None
    y = int(torch.round(coords[:, 0].float().mean()).item())
    x = int(torch.round(coords[:, 1].float().mean()).item())
    return y, x

def save_mask_visualization(args):
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples, _ = build_sample_index(
        args.data_root,
        metadata_path=args.metadata,
        file_index=args.file_index,
        max_files=args.max_files,
        max_samples=args.max_samples,
    )
    if not samples:
        raise ValueError(f"No PKL samples found under {args.data_root}")

    dataset = DiffractionPatternDataset(samples, image_size=args.image_size)
    chosen = sample_indices(dataset, args.num_examples, args.selection_mode, args.seed, args.center_exclude_radius)
    if not chosen:
        raise ValueError("No samples selected for visualization")

    fig, axes = plt.subplots(len(chosen), 4, figsize=(12, 3 * len(chosen)))
    if len(chosen) == 1:
        axes = axes[None, :]

    titles = ["Original", "Masked input", "Masked spot only", "Binary mask"]
    for row, idx in enumerate(chosen):
        item = dataset[idx]
        image = item["image"]
        _, height, width = image.shape
        mask = make_visual_mask(image, args)
        center_yx = mask_center(mask)
        masked_spot = image * mask

        arrays = [image[0], image[0].clone(), masked_spot[0], mask[0]]
        display_vmax = max(float(np.percentile(to_numpy(image[0]), args.display_percentile)), 1e-6)
        gray_value = float(args.mask_fill_value) * display_vmax
        arrays[1] = image[0] * (1.0 - mask[0]) + gray_value * mask[0]
        for col, arr in enumerate(arrays):
            vmax = None if col == 3 else display_vmax
            axes[row, col].imshow(to_numpy(arr), cmap="gray", vmin=0.0, vmax=vmax)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(titles[col])

        label = Path(item["path"]).name
        sample_index = item["index"]
        if sample_index >= 0:
            label += f" | sample {sample_index}"
        richness = diffraction_richness(image, args.center_exclude_radius)
        if center_yx is None:
            spot_text = "spot=not found"
        else:
            y0, x0 = center_yx
            spot_text = f"spot=({x0},{y0}), rich={richness:.3f}"
        axes[row, 0].set_ylabel(
            f"{height}x{width}\n{spot_text}\n{label}",
            fontsize=8,
        )

    fig.suptitle(
        f"Spot mask: selection={args.spot_selection}, spots={args.min_spots}-{args.max_spots}, "
        f"spot_radius={args.spot_radius}, center_exclude={args.center_exclude_radius}, "
        f"gray_display={args.mask_fill_value}, display_p={args.display_percentile}"
    )
    plt.tight_layout()
    out_path = args.output_dir / "mask_visualization.png"
    plt.savefig(out_path, dpi=200)
    print(f"saved: {out_path}")
    if args.show_window:
        plt.show()
    else:
        plt.close(fig)


def main():
    args = build_arg_parser().parse_args()
    save_mask_visualization(args)


if __name__ == "__main__":
    main()
