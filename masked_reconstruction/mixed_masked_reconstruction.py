import argparse
import csv
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from dataset import DiffractionPatternDataset, build_sample_index, split_samples
from masked_reconstruction import collate_batch, prepare_input
from metrics import masked_mae, masked_mse, psnr_from_mse, simple_ssim, weighted_masked_mse
from models import SmallUNet


SCRIPT_DEFAULTS = {
    "experimental_root": Path(r"G:\4D-ImageNet\exp"),
    "simulation_root": Path(r"G:\4D-ImageNet\sim"),
    "experimental_file_pattern": "*_master.pkl",
    "simulation_file_pattern": "*.pkl",
    "output_dir": Path(r"G:\4D-ImageNet\encoder\outputs\mixed_exp_sim_v1"),
    "max_experimental_samples": 5000,
    "max_simulation_samples": 5000,
    "mask_type": "spot",
    "mask_ratio": 0.5,
    "patch_size": 16,
    "spot_radius": 0.035,
    "center_exclude_radius": 0.06,
    "spot_rank": 1,
    "peak_window": 11,
    "spot_selection": "random",
    "min_spots": 1,
    "max_spots": 4,
    "min_peak_fraction": 0.12,
    "batch_size": 8,
    "gradient_accumulation_steps": 1,
    "lr": 1e-3,
    "epochs": 60,
    "early_stopping_patience": 12,
    "min_epochs": 25,
    "image_size": 192,
    "split": "0.7,0.15,0.15",
    "seed": 42,
    "preload_limit": 20000,
    "num_workers": 0,
    "base_channels": 16,
    "device": "auto",
    "amp": True,
    "compute_ssim": False,
    "display_percentile": 99.5,
    "loss_intensity_weight": 8.0,
    "loss_intensity_gamma": 1.0,
    "max_grad_norm": 1.0,
}


def parse_split(value: str) -> Tuple[float, float, float]:
    parts = tuple(float(x) for x in value.split(","))
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Split must contain train,val,test ratios")
    if sum(parts) <= 0:
        raise argparse.ArgumentTypeError("Split ratios must sum to a positive value")
    return parts


def next_available_output_dir(requested: Path) -> Path:
    if not requested.exists() or not any(requested.iterdir()):
        return requested
    match = re.fullmatch(r"(.+)_v(\d+)", requested.name)
    prefix = match.group(1) if match else requested.name
    version = int(match.group(2)) + 1 if match else 2
    while True:
        candidate = requested.parent / f"{prefix}_v{version}"
        if not candidate.exists() or not any(candidate.iterdir()):
            return candidate
        version += 1


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _material_from_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).parts[0]
    except (ValueError, IndexError):
        return path.parent.name


def annotate_domain(samples: Iterable[dict], domain: str, root: Path) -> List[dict]:
    annotated = []
    for source in samples:
        sample = dict(source)
        path = Path(sample["path"])
        sample["domain"] = domain
        sample["material"] = _material_from_path(path, root)
        sample["metadata_id"] = f"{domain}::{sample['metadata_id']}"
        annotated.append(sample)
    return annotated


def _split_group_counts(n_groups: int, ratios: Tuple[float, float, float]) -> Tuple[int, int, int]:
    if n_groups < 3:
        raise ValueError("Each experimental material needs at least three acquisition groups for train/val/test")
    total = sum(ratios)
    _, val_ratio, test_ratio = (value / total for value in ratios)
    n_val = max(1, int(round(val_ratio * n_groups)))
    n_test = max(1, int(round(test_ratio * n_groups)))
    while n_val + n_test >= n_groups:
        if n_val >= n_test and n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            break
    return n_groups - n_val - n_test, n_val, n_test


def split_experimental_by_material(
    samples: Sequence[dict],
    ratios: Tuple[float, float, float],
    seed: int,
) -> Tuple[List[dict], List[dict], List[dict]]:
    by_material: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        by_material[sample["material"]][sample["metadata_id"]].append(sample)

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for material in sorted(by_material):
        groups = list(by_material[material].items())
        rng.shuffle(groups)
        n_train, n_val, _ = _split_group_counts(len(groups), ratios)
        assignments = {
            "train": groups[:n_train],
            "val": groups[n_train:n_train + n_val],
            "test": groups[n_train + n_val:],
        }
        for split_name, assigned_groups in assignments.items():
            for _, group_samples in assigned_groups:
                splits[split_name].extend(group_samples)
    for values in splits.values():
        rng.shuffle(values)
    return splits["train"], splits["val"], splits["test"]


def make_dataset(samples: Sequence[dict], args) -> DiffractionPatternDataset:
    return DiffractionPatternDataset(
        samples,
        image_size=args.image_size,
        preload_limit=args.preload_limit,
    )


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)


def run_epoch(model, loader, optimizer, scaler, args, device, train: bool, accumulation_steps: int) -> float:
    model.train(train)
    total_loss = 0.0
    total_count = 0
    if train:
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, start=1):
        images, mask, _, model_input = prepare_input(batch, args, device)
        with torch.set_grad_enabled(train):
            with _autocast(device, args.amp and device.type == "cuda"):
                pred = model(model_input)
                loss = weighted_masked_mse(
                    pred,
                    images,
                    mask,
                    intensity_weight=args.loss_intensity_weight,
                    intensity_gamma=args.loss_intensity_gamma,
                )
            if train:
                scaler.scale(loss / accumulation_steps).backward()
                if step % accumulation_steps == 0 or step == len(loader):
                    if args.max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

        batch_size = images.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_count += batch_size
    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate(model, loader, args, device) -> dict:
    model.eval()
    totals = defaultdict(float)
    total_count = 0
    for batch in loader:
        images, mask, _, model_input = prepare_input(batch, args, device)
        with _autocast(device, args.amp and device.type == "cuda"):
            pred = model(model_input)
            values = {
                "masked_mse": masked_mse(pred, images, mask),
                "masked_mae": masked_mae(pred, images, mask),
                "weighted_masked_mse": weighted_masked_mse(
                    pred,
                    images,
                    mask,
                    intensity_weight=args.loss_intensity_weight,
                    intensity_gamma=args.loss_intensity_gamma,
                ),
            }
            if args.compute_ssim:
                values["ssim"] = simple_ssim(pred.clamp(0, 1), images)
        batch_size = images.shape[0]
        for name, value in values.items():
            totals[name] += float(value.detach().cpu()) * batch_size
        total_count += batch_size

    metrics = {name: value / max(total_count, 1) for name, value in totals.items()}
    metrics["psnr"] = psnr_from_mse(metrics["masked_mse"]) if total_count else float("nan")
    if "ssim" not in metrics:
        metrics["ssim"] = float("nan")
    metrics["num_samples"] = total_count
    return metrics


def memory_probe(model, loader, optimizer, scaler, args, device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    batch = next(iter(loader))
    images, mask, _, model_input = prepare_input(batch, args, device)
    optimizer.zero_grad(set_to_none=True)
    with _autocast(device, args.amp and device.type == "cuda"):
        pred = model(model_input)
        loss = weighted_masked_mse(
            pred,
            images,
            mask,
            intensity_weight=args.loss_intensity_weight,
            intensity_gamma=args.loss_intensity_gamma,
        )
    scaler.scale(loss).backward()
    optimizer.zero_grad(set_to_none=True)
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.reset_running_stats()
    del batch, images, mask, model_input, pred, loss
    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
        peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
        torch.cuda.empty_cache()
        print(f"CUDA memory probe: peak allocated={peak_allocated:.2f} GiB, reserved={peak_reserved:.2f} GiB")


def write_rows(path: Path, rows: Sequence[dict]):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_split_records(split_map: Dict[str, Sequence[dict]], output_dir: Path):
    summary = []
    manifest = []
    for split_name, samples in split_map.items():
        by_group: Dict[Tuple[str, str, str], int] = defaultdict(int)
        for sample in samples:
            key = (sample["domain"], sample["material"], sample["metadata_id"])
            by_group[key] += 1
        domains = sorted({sample["domain"] for sample in samples})
        for domain in domains:
            domain_samples = [sample for sample in samples if sample["domain"] == domain]
            summary.append(
                {
                    "split": split_name,
                    "domain": domain,
                    "num_samples": len(domain_samples),
                    "num_groups": len({sample["metadata_id"] for sample in domain_samples}),
                    "materials": ";".join(sorted({sample["material"] for sample in domain_samples})),
                }
            )
        for (domain, material, metadata_id), count in sorted(by_group.items()):
            manifest.append(
                {
                    "split": split_name,
                    "domain": domain,
                    "material": material,
                    "metadata_id": metadata_id,
                    "num_samples": count,
                }
            )
    write_rows(output_dir / "split_summary.csv", summary)
    write_rows(output_dir / "split_groups.csv", manifest)


def save_config(args, output_dir: Path, device: torch.device, batch_size: int, accumulation_steps: int):
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(
        {
            "resolved_device": str(device),
            "resolved_batch_size": batch_size,
            "resolved_gradient_accumulation_steps": accumulation_steps,
            "effective_batch_size": batch_size * accumulation_steps,
        }
    )
    with open(output_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


def representative_experimental_samples(samples: Sequence[dict], max_examples: int = 6) -> List[dict]:
    selected = []
    for material in sorted({sample["material"] for sample in samples}):
        candidates = [sample for sample in samples if sample["material"] == material]
        selected.append(candidates[len(candidates) // 2])
    return selected[:max_examples]


@torch.no_grad()
def save_experimental_examples(model, dataset, samples: Sequence[dict], args, device, output_dir: Path):
    selected = representative_experimental_samples(samples)
    if not selected:
        return
    set_seed(args.seed + 1000)
    sample_lookup = {
        (str(sample["path"]), int(sample["index"])): index
        for index, sample in enumerate(dataset.samples)
    }
    items = [
        dataset[sample_lookup[(str(sample["path"]), int(sample["index"]))]]
        for sample in selected
    ]
    batch = collate_batch(items)
    images, mask, masked, model_input = prepare_input(batch, args, device)
    with _autocast(device, args.amp and device.type == "cuda"):
        pred = model(model_input).clamp(0, 1)
    reconstruction = masked * (1.0 - mask) + pred * mask
    error = (pred - images).abs() * mask

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"]})
    n = len(selected)
    fig, axes = plt.subplots(n, 4, figsize=(8.2, 1.9 * n), squeeze=False)
    titles = ["Original experiment", "Masked input", "Reconstruction", "Masked error"]
    for row, sample in enumerate(selected):
        reference = images[row, 0].detach().cpu().numpy()
        display_vmax = max(float(np.percentile(reference, args.display_percentile)), 1e-6)
        arrays = [images[row, 0], masked[row, 0], reconstruction[row, 0], error[row, 0]]
        for column, array in enumerate(arrays):
            axes[row, column].imshow(array.detach().float().cpu().numpy(), cmap="gray", vmin=0, vmax=display_vmax)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            if row == 0:
                axes[row, column].set_title(titles[column], fontsize=9)
        axes[row, 0].set_ylabel(sample["material"], fontsize=9, rotation=90, labelpad=8)
    fig.tight_layout(pad=0.35, w_pad=0.2, h_pad=0.2)
    fig.savefig(output_dir / "experimental_reconstruction_examples.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for sample in selected:
        rows.append(
            {
                "material": sample["material"],
                "path": str(sample["path"]),
                "pattern_index": sample["index"],
                "metadata_id": sample["metadata_id"],
            }
        )
    write_rows(output_dir / "experimental_reconstruction_examples.csv", rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mixed experimental/simulation masked reconstruction")
    parser.add_argument("--experimental-root", type=Path, default=SCRIPT_DEFAULTS["experimental_root"])
    parser.add_argument("--simulation-root", type=Path, default=SCRIPT_DEFAULTS["simulation_root"])
    parser.add_argument("--experimental-file-pattern", default=SCRIPT_DEFAULTS["experimental_file_pattern"])
    parser.add_argument("--simulation-file-pattern", default=SCRIPT_DEFAULTS["simulation_file_pattern"])
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DEFAULTS["output_dir"])
    parser.add_argument("--max-experimental-samples", type=int, default=SCRIPT_DEFAULTS["max_experimental_samples"])
    parser.add_argument("--max-simulation-samples", type=int, default=SCRIPT_DEFAULTS["max_simulation_samples"])
    parser.add_argument("--mask-type", choices=["patch", "wedge", "central", "spot"], default=SCRIPT_DEFAULTS["mask_type"])
    parser.add_argument("--mask-ratio", type=float, default=SCRIPT_DEFAULTS["mask_ratio"])
    parser.add_argument("--patch-size", type=int, default=SCRIPT_DEFAULTS["patch_size"])
    parser.add_argument("--spot-radius", type=float, default=SCRIPT_DEFAULTS["spot_radius"])
    parser.add_argument("--center-exclude-radius", type=float, default=SCRIPT_DEFAULTS["center_exclude_radius"])
    parser.add_argument("--spot-rank", type=int, default=SCRIPT_DEFAULTS["spot_rank"])
    parser.add_argument("--peak-window", type=int, default=SCRIPT_DEFAULTS["peak_window"])
    parser.add_argument("--spot-selection", choices=["random", "nearest_bright", "brightest"], default=SCRIPT_DEFAULTS["spot_selection"])
    parser.add_argument("--min-spots", type=int, default=SCRIPT_DEFAULTS["min_spots"])
    parser.add_argument("--max-spots", type=int, default=SCRIPT_DEFAULTS["max_spots"])
    parser.add_argument("--min-peak-fraction", type=float, default=SCRIPT_DEFAULTS["min_peak_fraction"])
    parser.add_argument("--batch-size", type=int, default=SCRIPT_DEFAULTS["batch_size"])
    parser.add_argument("--gradient-accumulation-steps", type=int, default=SCRIPT_DEFAULTS["gradient_accumulation_steps"])
    parser.add_argument("--lr", type=float, default=SCRIPT_DEFAULTS["lr"])
    parser.add_argument("--epochs", type=int, default=SCRIPT_DEFAULTS["epochs"])
    parser.add_argument("--early-stopping-patience", type=int, default=SCRIPT_DEFAULTS["early_stopping_patience"])
    parser.add_argument("--min-epochs", type=int, default=SCRIPT_DEFAULTS["min_epochs"])
    parser.add_argument("--image-size", type=int, default=SCRIPT_DEFAULTS["image_size"])
    parser.add_argument("--split", type=parse_split, default=parse_split(SCRIPT_DEFAULTS["split"]))
    parser.add_argument("--seed", type=int, default=SCRIPT_DEFAULTS["seed"])
    parser.add_argument("--preload-limit", type=int, default=SCRIPT_DEFAULTS["preload_limit"])
    parser.add_argument("--num-workers", type=int, default=SCRIPT_DEFAULTS["num_workers"])
    parser.add_argument("--base-channels", type=int, default=SCRIPT_DEFAULTS["base_channels"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=SCRIPT_DEFAULTS["device"])
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=SCRIPT_DEFAULTS["amp"])
    parser.add_argument("--compute-ssim", action=argparse.BooleanOptionalAction, default=SCRIPT_DEFAULTS["compute_ssim"])
    parser.add_argument("--display-percentile", type=float, default=SCRIPT_DEFAULTS["display_percentile"])
    parser.add_argument("--loss-intensity-weight", type=float, default=SCRIPT_DEFAULTS["loss_intensity_weight"])
    parser.add_argument("--loss-intensity-gamma", type=float, default=SCRIPT_DEFAULTS["loss_intensity_gamma"])
    parser.add_argument("--max-grad-norm", type=float, default=SCRIPT_DEFAULTS["max_grad_norm"])
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty output directory")
    return parser


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)
    if not args.overwrite:
        resolved_output_dir = next_available_output_dir(args.output_dir)
        if resolved_output_dir != args.output_dir:
            print(f"Output directory already contains a run; using {resolved_output_dir} instead")
        args.output_dir = resolved_output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"Using {torch.cuda.get_device_name(device)} with automatic mixed precision={args.amp}")
    else:
        args.amp = False
        print("Using CPU")

    experimental, _ = build_sample_index(
        args.experimental_root,
        max_samples=args.max_experimental_samples,
        file_pattern=args.experimental_file_pattern,
    )
    simulation, _ = build_sample_index(
        args.simulation_root,
        max_samples=args.max_simulation_samples,
        file_pattern=args.simulation_file_pattern,
    )
    experimental = annotate_domain(experimental, "experimental", args.experimental_root)
    simulation = annotate_domain(simulation, "simulation", args.simulation_root)

    exp_train, exp_val, exp_test = split_experimental_by_material(experimental, args.split, args.seed)
    sim_train, sim_val, sim_test = split_samples(simulation, args.split, args.seed, True)
    rng = random.Random(args.seed)
    train_samples = exp_train + sim_train
    val_samples = exp_val + sim_val
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    split_map = {
        "train": train_samples,
        "validation": val_samples,
        "experimental_test": exp_test,
        "simulation_test": sim_test,
    }
    save_split_records(split_map, args.output_dir)
    for split_name, split_values in split_map.items():
        domains = {domain: sum(sample["domain"] == domain for sample in split_values) for domain in ("experimental", "simulation")}
        print(f"{split_name}: total={len(split_values)}, experimental={domains['experimental']}, simulation={domains['simulation']}")

    train_dataset = make_dataset(train_samples, args)
    val_dataset = make_dataset(val_samples, args)
    exp_test_dataset = make_dataset(exp_test, args)
    sim_test_dataset = make_dataset(sim_test, args)

    requested_batch_size = max(1, args.batch_size)
    batch_size = requested_batch_size
    accumulation_steps = max(1, args.gradient_accumulation_steps)
    model = SmallUNet(in_channels=2, out_channels=1, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    while True:
        train_loader = make_loader(train_dataset, batch_size, True, args.num_workers, args.seed)
        try:
            memory_probe(model, train_loader, optimizer, scaler, args, device)
            break
        except torch.OutOfMemoryError:
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            accumulation_steps = max(
                accumulation_steps,
                math.ceil(requested_batch_size * args.gradient_accumulation_steps / batch_size),
            )
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            print(f"CUDA OOM during probe; retrying with batch_size={batch_size}, accumulation={accumulation_steps}")

    val_loader = make_loader(val_dataset, batch_size, False, args.num_workers, args.seed + 1)
    exp_test_loader = make_loader(exp_test_dataset, batch_size, False, args.num_workers, args.seed + 2)
    sim_test_loader = make_loader(sim_test_dataset, batch_size, False, args.num_workers, args.seed + 3)
    save_config(args, args.output_dir, device, batch_size, accumulation_steps)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5)
    history = []
    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_path = args.output_dir / "model_best_mixed.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, scaler, args, device, True, accumulation_steps)
        val_loss = run_epoch(model, val_loader, optimizer, scaler, args, device, False, accumulation_steps)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": current_lr})
        print(f"epoch {epoch:03d}: train={train_loss:.6f} val={val_loss:.6f} lr={current_lr:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "best_val_loss": best_val,
                    "best_epoch": best_epoch,
                    "training_domains": ["experimental", "simulation"],
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1

        write_rows(args.output_dir / "train_val_loss.csv", history)
        if epoch >= args.min_epochs and epochs_without_improvement >= args.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}")
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    torch.save(
        {
            "model_state": model.state_dict(),
            "args": vars(args),
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
            "training_domains": ["experimental", "simulation"],
        },
        args.output_dir / "model_final_mixed.pt",
    )

    exp_metrics = evaluate(model, exp_test_loader, args, device)
    sim_metrics = evaluate(model, sim_test_loader, args, device)
    write_rows(args.output_dir / "metrics_experimental_test.csv", [exp_metrics])
    write_rows(args.output_dir / "metrics_simulation_test.csv", [sim_metrics])

    epochs = [row["epoch"] for row in history]
    fig, axis = plt.subplots(figsize=(5.2, 3.6))
    axis.plot(epochs, [row["train_loss"] for row in history], label="Mixed-domain train")
    axis.plot(epochs, [row["val_loss"] for row in history], label="Mixed-domain validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Weighted masked MSE")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "loss_curve.png", dpi=250)
    plt.close(fig)

    save_experimental_examples(model, exp_test_dataset, exp_test, args, device, args.output_dir)
    print(f"Best epoch: {best_epoch}; best validation loss: {best_val:.6f}")
    print("Experimental test metrics:", exp_metrics)
    print("Simulation test metrics:", sim_metrics)
    print(f"Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
