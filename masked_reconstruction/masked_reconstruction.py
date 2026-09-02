import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from dataset import DiffractionPatternDataset, build_sample_index, split_samples
from masks import make_mask
from metrics import masked_mae, masked_mse, psnr_from_mse, simple_ssim, weighted_masked_mse
from models import SmallUNet


SCRIPT_DEFAULTS = {
    "data_root": Path(r"G:\4D-ImageNet\sim"),
    "metadata": None,
    "file_index": None,
    "output_dir": Path(r"G:\4D-ImageNet\encoder\outputs"),
    "mask_type": "spot",
    "mask_ratio": 0.5,  # used by patch/wedge/central masks
    "patch_size": 16,
    "spot_radius": 0.035,
    "center_exclude_radius": 0.06,
    "spot_rank": 1,
    "peak_window": 11,
    "spot_selection": "random",  # random, nearest_bright, or brightest
    "min_spots": 1,
    "max_spots": 4,
    "min_peak_fraction": 0.12,
    "batch_size": 8,
    "lr": 1e-3,
    "epochs": 100,
    "image_size": 256,
    "split": "0.7,0.15,0.15",
    "seed": 42,
    "max_files": None,  # None uses all PKL files under data_root.
    "max_samples": 10000,  # None uses all patterns; an int samples evenly across PKL files.
    "preload_limit": 20000,  # preload spl89it samples if split size is <= this value; 0 disables preloading.
    "num_workers": 0,
    "base_channels": 16,
    "device": "auto",
    "compute_ssim": False,
    "display_percentile": 99.5,
    "loss_intensity_weight": 8.0,
    "loss_intensity_gamma": 1.0,
}


def parse_split(value: str):
    parts = [float(x) for x in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Split ratio must have three comma-separated values, e.g. 0.7,0.15,0.15")
    return tuple(parts)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_batch(batch):
    images = [item["image"] for item in batch]
    channels = images[0].shape[0]
    max_h = max(int(img.shape[-2]) for img in images)
    max_w = max(int(img.shape[-1]) for img in images)
    padded = images[0].new_zeros((len(images), channels, max_h, max_w))
    valid_mask = images[0].new_zeros((len(images), 1, max_h, max_w))
    for i, img in enumerate(images):
        h, w = int(img.shape[-2]), int(img.shape[-1])
        padded[i, :, :h, :w] = img
        valid_mask[i, :, :h, :w] = 1.0
    return {
        "image": padded,
        "valid_mask": valid_mask,
        "path": [item["path"] for item in batch],
        "index": torch.tensor([item["index"] for item in batch], dtype=torch.long),
        "metadata_id": [item["metadata_id"] for item in batch],
    }


def make_loader(samples, image_size, batch_size, shuffle, num_workers=0, preload_limit=2000):
    dataset = DiffractionPatternDataset(samples, image_size=image_size, preload_limit=preload_limit)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_batch, pin_memory=torch.cuda.is_available())


def prepare_input(batch, args, device):
    images = batch["image"].to(device, non_blocking=True)
    valid_mask = batch.get("valid_mask")
    if valid_mask is not None:
        valid_mask = valid_mask.to(device, non_blocking=True)
    else:
        valid_mask = torch.ones_like(images[:, :1])
    _, _, h, w = images.shape
    mask = make_mask(
        images.shape[0],
        h,
        w,
        args.mask_type,
        args.mask_ratio,
        args.patch_size,
        device=device,
        images=images,
        valid_mask=valid_mask,
        spot_radius=args.spot_radius,
        center_exclude_radius=args.center_exclude_radius,
        spot_rank=args.spot_rank,
        peak_window=args.peak_window,
        spot_selection=args.spot_selection,
        min_spots=args.min_spots,
        max_spots=args.max_spots,
        min_peak_fraction=args.min_peak_fraction,
    )
    mask = mask * valid_mask
    masked = images * (1.0 - mask)
    model_input = torch.cat([masked, mask], dim=1)
    return images, mask, masked, model_input


def run_epoch(model, loader, optimizer, args, device, train: bool):
    model.train(train)
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        images, mask, _, model_input = prepare_input(batch, args, device)
        with torch.set_grad_enabled(train):
            pred = model(model_input)
            loss = weighted_masked_mse(
                pred,
                images,
                mask,
                intensity_weight=args.loss_intensity_weight,
                intensity_gamma=args.loss_intensity_gamma,
            )
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        bs = images.shape[0]
        total_loss += float(loss.detach().cpu()) * bs
        total_count += bs
    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate(model, loader, args, device):
    model.eval()
    mse_values = []
    mae_values = []
    weighted_mse_values = []
    ssim_values = []
    for batch in loader:
        images, mask, _, model_input = prepare_input(batch, args, device)
        pred = model(model_input)
        mse = masked_mse(pred, images, mask)
        mae = masked_mae(pred, images, mask)
        weighted_mse = weighted_masked_mse(
            pred,
            images,
            mask,
            intensity_weight=args.loss_intensity_weight,
            intensity_gamma=args.loss_intensity_gamma,
        )
        mse_values.append(float(mse.cpu()))
        mae_values.append(float(mae.cpu()))
        weighted_mse_values.append(float(weighted_mse.cpu()))
        if args.compute_ssim:
            ssim_values.append(float(simple_ssim(pred.clamp(0, 1), images).cpu()))
    mse_mean = float(np.mean(mse_values)) if mse_values else float("nan")
    mae_mean = float(np.mean(mae_values)) if mae_values else float("nan")
    weighted_mse_mean = float(np.mean(weighted_mse_values)) if weighted_mse_values else float("nan")
    return {
        "masked_mse": mse_mean,
        "masked_mae": mae_mean,
        "weighted_masked_mse": weighted_mse_mean,
        "psnr": psnr_from_mse(mse_mean) if np.isfinite(mse_mean) else float("nan"),
        "ssim": float(np.mean(ssim_values)) if ssim_values else float("nan"),
    }


def save_loss_csv(history, output_dir: Path):
    with open(output_dir / "train_val_loss.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)


def save_metrics_csv(metrics, output_dir: Path):
    with open(output_dir / "metrics_test.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def save_split_summary(train_samples, val_samples, test_samples, output_dir: Path):
    rows = []
    for split_name, samples in (("train", train_samples), ("val", val_samples), ("test", test_samples)):
        metadata_ids = sorted({s["metadata_id"] for s in samples})
        rows.append({"split": split_name, "num_samples": len(samples), "num_metadata_ids": len(metadata_ids)})
    with open(output_dir / "split_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "num_samples", "num_metadata_ids"])
        writer.writeheader()
        writer.writerows(rows)


def save_loss_plot(history, output_dir: Path):
    epochs = [row["epoch"] for row in history]
    train = [row["train_loss"] for row in history]
    val = [row["val_loss"] for row in history]
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train, label="train")
    plt.plot(epochs, val, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Masked MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=200)
    plt.close()


@torch.no_grad()
def save_examples(model, loader, args, device, output_dir: Path, max_examples: int = 4):
    model.eval()
    try:
        batch = next(iter(loader))
    except StopIteration:
        return
    images, mask, masked, model_input = prepare_input(batch, args, device)
    pred = model(model_input).clamp(0, 1)
    reconstruction = masked * (1.0 - mask) + pred * mask
    error = (pred - images).abs() * mask
    n = min(max_examples, images.shape[0])
    fig, axes = plt.subplots(n, 4, figsize=(10, 2.5 * n))
    if n == 1:
        axes = axes[None, :]
    titles = ["Original", "Masked input", "Reconstruction", "Masked error"]
    display_percentile = float(getattr(args, "display_percentile", 99.5))
    for i in range(n):
        arrays = [images[i, 0], masked[i, 0], reconstruction[i, 0], error[i, 0]]
        reference = arrays[0].detach().cpu().numpy()
        display_vmax = max(float(np.percentile(reference, display_percentile)), 1e-6)
        for j, arr in enumerate(arrays):
            axes[i, j].imshow(arr.detach().cpu().numpy(), cmap="gray", vmin=0.0, vmax=display_vmax)
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(titles[j])
    plt.tight_layout()
    plt.savefig(output_dir / "reconstruction_examples.png", dpi=200)
    plt.close()


def write_config(args, output_dir: Path, device: torch.device):
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["device"] = str(device)
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Masked diffraction-pattern reconstruction baseline")
    parser.add_argument("--data-root", default=SCRIPT_DEFAULTS["data_root"], type=Path)
    parser.add_argument("--metadata", default=SCRIPT_DEFAULTS["metadata"], type=Path)
    parser.add_argument("--file-index", default=SCRIPT_DEFAULTS["file_index"], type=Path)
    parser.add_argument("--output-dir", default=SCRIPT_DEFAULTS["output_dir"], type=Path)
    parser.add_argument("--mask-type", default=SCRIPT_DEFAULTS["mask_type"], choices=["patch", "wedge", "central", "spot"])
    parser.add_argument("--mask-ratio", default=SCRIPT_DEFAULTS["mask_ratio"], type=float)
    parser.add_argument("--patch-size", default=SCRIPT_DEFAULTS["patch_size"], type=int)
    parser.add_argument("--spot-radius", default=SCRIPT_DEFAULTS["spot_radius"], type=float)
    parser.add_argument("--center-exclude-radius", default=SCRIPT_DEFAULTS["center_exclude_radius"], type=float)
    parser.add_argument("--spot-rank", default=SCRIPT_DEFAULTS["spot_rank"], type=int)
    parser.add_argument("--peak-window", default=SCRIPT_DEFAULTS["peak_window"], type=int)
    parser.add_argument("--spot-selection", default=SCRIPT_DEFAULTS["spot_selection"], choices=["random", "nearest_bright", "brightest"])
    parser.add_argument("--min-spots", default=SCRIPT_DEFAULTS["min_spots"], type=int)
    parser.add_argument("--max-spots", default=SCRIPT_DEFAULTS["max_spots"], type=int)
    parser.add_argument("--min-peak-fraction", default=SCRIPT_DEFAULTS["min_peak_fraction"], type=float)
    parser.add_argument("--batch-size", default=SCRIPT_DEFAULTS["batch_size"], type=int)
    parser.add_argument("--lr", default=SCRIPT_DEFAULTS["lr"], type=float)
    parser.add_argument("--epochs", default=SCRIPT_DEFAULTS["epochs"], type=int)
    parser.add_argument("--image-size", default=SCRIPT_DEFAULTS["image_size"], type=int)
    parser.add_argument("--split", default=parse_split(SCRIPT_DEFAULTS["split"]), type=parse_split)
    parser.add_argument("--seed", default=SCRIPT_DEFAULTS["seed"], type=int)
    parser.add_argument("--max-files", default=SCRIPT_DEFAULTS["max_files"], type=int)
    parser.add_argument("--max-samples", default=SCRIPT_DEFAULTS["max_samples"], type=int)
    parser.add_argument("--preload-limit", default=SCRIPT_DEFAULTS["preload_limit"], type=int)
    parser.add_argument("--num-workers", default=SCRIPT_DEFAULTS["num_workers"], type=int)
    parser.add_argument("--base-channels", default=SCRIPT_DEFAULTS["base_channels"], type=int)
    parser.add_argument("--device", default=SCRIPT_DEFAULTS["device"], choices=["auto", "cpu", "cuda"])
    parser.add_argument("--compute-ssim", action=argparse.BooleanOptionalAction, default=SCRIPT_DEFAULTS["compute_ssim"])
    parser.add_argument("--display-percentile", default=SCRIPT_DEFAULTS["display_percentile"], type=float)
    parser.add_argument("--loss-intensity-weight", default=SCRIPT_DEFAULTS["loss_intensity_weight"], type=float)
    parser.add_argument("--loss-intensity-gamma", default=SCRIPT_DEFAULTS["loss_intensity_gamma"], type=float)
    return parser


def main():
    args = build_arg_parser().parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    write_config(args, args.output_dir, device)

    samples, has_metadata_groups = build_sample_index(
        args.data_root,
        metadata_path=args.metadata,
        file_index=args.file_index,
        max_files=args.max_files,
        max_samples=args.max_samples,
    )
    train_samples, val_samples, test_samples = split_samples(samples, args.split, args.seed, has_metadata_groups)
    if not train_samples:
        raise ValueError("Training split is empty. Increase data size or adjust split ratios.")
    if not val_samples:
        val_samples = train_samples[: min(len(train_samples), max(1, args.batch_size))]
    if not test_samples:
        test_samples = val_samples
    save_split_summary(train_samples, val_samples, test_samples, args.output_dir)

    print(f"Split sizes: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")
    train_loader = make_loader(train_samples, args.image_size, args.batch_size, True, args.num_workers, args.preload_limit)
    val_loader = make_loader(val_samples, args.image_size, args.batch_size, False, args.num_workers, args.preload_limit)
    test_loader = make_loader(test_samples, args.image_size, args.batch_size, False, args.num_workers, args.preload_limit)

    model = SmallUNet(in_channels=2, out_channels=1, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, args, device, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, args, device, train=False)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch {epoch:03d}: train={train_loss:.6f} val={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict(), "args": vars(args), "best_val_loss": best_val}, args.output_dir / "model_best.pt")

    save_loss_csv(history, args.output_dir)
    save_loss_plot(history, args.output_dir)
    if (args.output_dir / "model_best.pt").exists():
        checkpoint = torch.load(args.output_dir / "model_best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
    metrics = evaluate(model, test_loader, args, device)
    save_metrics_csv(metrics, args.output_dir)
    save_examples(model, test_loader, args, device, args.output_dir)
    print("test metrics:", metrics)


if __name__ == "__main__":
    main()



