from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Iterator

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from configs import DEFAULT_OUTPUT_DIR, MODEL_CONFIGS, PROJECT_ROOT, WEIGHT_PATHS
from model import SupervisedDNSR


# VSCode Run and Debug can use this block directly. Command-line arguments are optional.
RUN_CONFIG = {
    "input": PROJECT_ROOT
    / "exp"
    / "Ag"
    / "Ag_FOV134nm_CL160mm_185_185_3ms_2.1mrad_920kx_000_master",
    "output_dir": DEFAULT_OUTPUT_DIR, 
    "version": "r4",
    "weight": None,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "confidence": 0.5,
    "max_samples": 8,
    "batch_size": 1,
    "image_size": 256,
    "preprocess": "log1p_max",
    "save_visualizations": True,
}


def _install_numpy_pickle_compat() -> None:
    """Support pkl files saved by NumPy 2.x when running under NumPy 1.x.

    NumPy 2 stores some pickle references under ``numpy._core``. Older
    environments expose the same modules as ``numpy.core`` instead.
    """
    try:
        import numpy.core as numpy_core
        import numpy.core.numeric as numpy_core_numeric
        import numpy.core.multiarray as numpy_core_multiarray
    except Exception:
        return

    sys.modules.setdefault("numpy._core", numpy_core)
    sys.modules.setdefault("numpy._core.numeric", numpy_core_numeric)
    sys.modules.setdefault("numpy._core.multiarray", numpy_core_multiarray)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _iter_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    files = []
    for suffix in ("*.pkl", "*.h5", "*.hdf5"):
        files.extend(p for p in input_path.rglob(suffix) if not p.name.startswith("._"))
    return sorted(files)


def _iter_pkl_patterns(path: Path) -> Iterator[dict[str, Any]]:
    _install_numpy_pickle_compat()
    with path.open("rb") as handle:
        payload = pickle.load(handle)

    if isinstance(payload, dict) and "diffraction_patterns" in payload:
        patterns = np.asarray(payload["diffraction_patterns"])
        for index in range(patterns.shape[0]):
            yield {
                "source_file": path,
                "index": index,
                "pattern": patterns[index],
                "metadata": {
                    "material": payload.get("material"),
                    "scan_position": (
                        payload.get("scan_positions")[index].tolist()
                        if payload.get("scan_positions") is not None
                        else None
                    ),
                    "estimated_beam_center": payload.get("estimated_beam_center"),
                },
            }
        return

    if isinstance(payload, dict) and "diffraction_pattern" in payload:
        yield {
            "source_file": path,
            "index": 0,
            "pattern": payload["diffraction_pattern"],
            "metadata": {k: v for k, v in payload.items() if k != "diffraction_pattern"},
        }
        return

    if isinstance(payload, dict) and "item_list" in payload:
        for index, item in enumerate(payload["item_list"]):
            if "dp" not in item:
                continue
            yield {
                "source_file": path,
                "index": index,
                "pattern": item["dp"],
                "metadata": {},
            }
        return

    raise ValueError(f"Unsupported pkl structure: {path}")


def _find_h5_dataset(handle: h5py.File, preferred_path: str | None = None) -> h5py.Dataset:
    if preferred_path:
        dataset = handle[preferred_path]
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError(f"H5 path is not a dataset: {preferred_path}")
        return dataset

    candidates: list[tuple[int, str, h5py.Dataset]] = []

    def visit(name: str, obj: h5py.Dataset) -> None:
        if isinstance(obj, h5py.Dataset) and obj.ndim >= 2 and np.issubdtype(obj.dtype, np.number):
            score = obj.ndim * 10 + int(np.prod(obj.shape[-2:]))
            candidates.append((score, name, obj))

    handle.visititems(visit)
    if not candidates:
        raise ValueError("No numeric 2D-or-higher dataset found in h5 file.")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][2]


def _iter_h5_patterns(path: Path, dataset_path: str | None = None) -> Iterator[dict[str, Any]]:
    with h5py.File(path, "r") as handle:
        dataset = _find_h5_dataset(handle, dataset_path)
        shape = dataset.shape
        if dataset.ndim == 2:
            yield {"source_file": path, "index": 0, "pattern": dataset[...], "metadata": {"h5_dataset": dataset.name}}
            return

        leading_shape = shape[:-2]
        total = int(np.prod(leading_shape))
        for flat_index in range(total):
            multi_index = np.unravel_index(flat_index, leading_shape)
            yield {
                "source_file": path,
                "index": flat_index,
                "pattern": dataset[multi_index],
                "metadata": {"h5_dataset": dataset.name, "h5_index": list(multi_index)},
            }


def iter_patterns(input_path: Path, h5_dataset: str | None = None) -> Iterator[dict[str, Any]]:
    for file_path in _iter_input_files(input_path):
        suffix = file_path.suffix.lower()
        if suffix == ".pkl":
            yield from _iter_pkl_patterns(file_path)
        elif suffix in {".h5", ".hdf5"}:
            yield from _iter_h5_patterns(file_path, h5_dataset)


def preprocess_pattern(pattern: np.ndarray, image_size: int, mode: str) -> tuple[torch.Tensor, np.ndarray]:
    image = np.asarray(pattern).squeeze().astype(np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected 2D diffraction pattern, got shape {image.shape}")

    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    image = np.maximum(image, 0.0)

    if mode == "log1p_max":
        processed = np.log1p(image)
        max_value = float(processed.max())
        if max_value > 0:
            processed = processed / max_value
    elif mode == "sum":
        total = float(image.sum())
        processed = image / total if total > 0 else image
    elif mode == "max":
        max_value = float(image.max())
        processed = image / max_value if max_value > 0 else image
    else:
        raise ValueError(f"Unknown preprocess mode: {mode}")

    tensor = torch.from_numpy(processed.astype(np.float32))[None, None]
    if tensor.shape[-2:] != (image_size, image_size):
        tensor = F.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return tensor[0], image


def build_model(version: str, weight_path: Path, device: torch.device) -> SupervisedDNSR:
    model = SupervisedDNSR(MODEL_CONFIGS[version]).to(device)
    try:
        state = torch.load(weight_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(weight_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"Loaded with missing keys={len(missing)}, unexpected keys={len(unexpected)}")
        if unexpected:
            print(f"Unexpected keys sample: {unexpected[:5]}")
        if missing:
            print(f"Missing keys sample: {missing[:5]}")
    model.eval()
    return model


@torch.no_grad()
def run_inference(config: dict[str, Any], h5_dataset: str | None = None) -> Path:
    input_path = Path(config["input"]).resolve()
    output_dir = Path(config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    version = config["version"]
    weight_path = Path(config["weight"]).resolve() if config["weight"] else WEIGHT_PATHS[version]
    device = torch.device(config["device"])
    model = build_model(version, weight_path, device)

    results_path = output_dir / f"disk_predictions_{version}.jsonl"
    visual_dir = output_dir / "visualizations"
    if config["save_visualizations"]:
        visual_dir.mkdir(parents=True, exist_ok=True)

    samples = iter_patterns(input_path, h5_dataset=h5_dataset)
    pending: list[dict[str, Any]] = []
    written = 0

    with results_path.open("w", encoding="utf-8") as handle:
        progress = tqdm(total=config["max_samples"], desc="Disk detection")
        for sample in samples:
            tensor, display_image = preprocess_pattern(sample["pattern"], config["image_size"], config["preprocess"])
            sample["tensor"] = tensor
            sample["display_image"] = display_image
            pending.append(sample)

            if len(pending) < config["batch_size"]:
                continue

            written += _process_batch(model, pending, config, device, handle, visual_dir)
            progress.update(len(pending))
            pending = []
            if written >= config["max_samples"]:
                break

        if pending and written < config["max_samples"]:
            written += _process_batch(model, pending, config, device, handle, visual_dir)
            progress.update(len(pending))
        progress.close()

    print(f"Saved {written} prediction records to {results_path}")
    return results_path


def _process_batch(
    model: SupervisedDNSR,
    batch: list[dict[str, Any]],
    config: dict[str, Any],
    device: torch.device,
    handle,
    visual_dir: Path,
) -> int:
    records = predict_batch_records(model, batch, config, device)

    for item, record in zip(batch, records):
        handle.write(json.dumps(record, ensure_ascii=False, default=_jsonable) + "\n")

        if config["save_visualizations"]:
            name = f"{Path(item['source_file']).stem}_{int(item['index']):05d}.png"
            save_overlay(item["display_image"], record["disks"], record["center_beam"], visual_dir / name)

    return len(records)


@torch.no_grad()
def predict_batch_records(
    model: SupervisedDNSR,
    batch: list[dict[str, Any]],
    config: dict[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    """Return JSON-ready detector records for one preprocessed batch."""
    image_batch = torch.stack([item["tensor"] for item in batch]).to(device)
    outputs = model(image_batch)[-1]
    probabilities = outputs["pred_logits"].softmax(-1)[:, :, :-1]
    scores = probabilities.max(dim=-1).values
    coords = outputs["pred_coords"]
    centers = outputs["pred_center_beam"]
    records: list[dict[str, Any]] = []

    for batch_index, item in enumerate(batch):
        height, width = item["display_image"].shape
        keep = scores[batch_index] > float(config["confidence"])
        kept_coords = coords[batch_index][keep].detach().cpu().numpy()
        kept_scores = scores[batch_index][keep].detach().cpu().numpy()
        center = centers[batch_index].detach().cpu().numpy()

        disks = [
            {
                "x": float(x * width),
                "y": float(y * height),
                "x_norm": float(x),
                "y_norm": float(y),
                "score": float(score),
            }
            for (x, y), score in zip(kept_coords, kept_scores)
        ]
        record = {
            "source_file": str(item["source_file"]),
            "index": int(item["index"]),
            "original_shape": [int(height), int(width)],
            "version": config["version"],
            "weight": str(config["weight"] or WEIGHT_PATHS[config["version"]]),
            "confidence": float(config["confidence"]),
            "center_beam": {
                "x": float(center[0] * width),
                "y": float(center[1] * height),
                "x_norm": float(center[0]),
                "y_norm": float(center[1]),
            },
            "disk_count": len(disks),
            "disks": disks,
            "metadata": _jsonable(item.get("metadata", {})),
        }
        records.append(record)

    return records


def save_overlay(image: np.ndarray, disks: list[dict[str, float]], center: dict[str, float], save_path: Path) -> None:
    display = np.log1p(np.maximum(image.astype(np.float32), 0.0))
    plt.figure(figsize=(6, 6))
    plt.imshow(display, cmap="gray")
    if disks:
        plt.scatter(
            [disk["x"] for disk in disks],
            [disk["y"] for disk in disks],
            marker="x",
            s=36,
            linewidths=1.2,
            c="#00d2ff",
        )
    plt.scatter(center["x"], center["y"], marker="+", s=80, linewidths=1.5, c="#ffd000")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(save_path, dpi=160, bbox_inches="tight", pad_inches=0)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run disk detection on 4D-ImageNet pkl/h5 diffraction patterns.")
    parser.add_argument("--input", type=Path, default=RUN_CONFIG["input"])
    parser.add_argument("--output-dir", type=Path, default=RUN_CONFIG["output_dir"])
    parser.add_argument("--version", choices=sorted(MODEL_CONFIGS), default=RUN_CONFIG["version"])
    parser.add_argument("--weight", type=Path, default=RUN_CONFIG["weight"])
    parser.add_argument("--device", default=RUN_CONFIG["device"])
    parser.add_argument("--confidence", type=float, default=RUN_CONFIG["confidence"])
    parser.add_argument("--max-samples", type=int, default=RUN_CONFIG["max_samples"])
    parser.add_argument("--batch-size", type=int, default=RUN_CONFIG["batch_size"])
    parser.add_argument("--image-size", type=int, default=RUN_CONFIG["image_size"])
    parser.add_argument("--preprocess", choices=["log1p_max", "sum", "max"], default=RUN_CONFIG["preprocess"])
    parser.add_argument("--h5-dataset", default=None)
    parser.add_argument("--no-visualizations", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = dict(RUN_CONFIG)
    config.update(
        {
            "input": args.input,
            "output_dir": args.output_dir,
            "version": args.version,
            "weight": args.weight,
            "device": args.device,
            "confidence": args.confidence,
            "max_samples": args.max_samples,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "preprocess": args.preprocess,
            "save_visualizations": not args.no_visualizations,
        }
    )
    run_inference(config, h5_dataset=args.h5_dataset)
