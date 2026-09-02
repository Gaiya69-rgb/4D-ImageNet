from __future__ import annotations

import argparse
import gc
import json
from itertools import chain
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import torch
from tqdm import tqdm

from configs import MODEL_CONFIGS, PROJECT_ROOT, WEIGHT_PATHS
from infer_4d_imagenet import (
    _iter_pkl_patterns,
    _jsonable,
    build_model,
    predict_batch_records,
    preprocess_pattern,
)


# This block is the primary interface for VSCode Run and Debug.
RUN_CONFIG = {
    "input_root": PROJECT_ROOT / "exp",
    "output_root": PROJECT_ROOT / "label",
    "version": "r4",
    "weight": None,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "confidence": 0.5,
    "batch_size": 32,
    "image_size": 256,
    "preprocess": "log1p_max",
    "overwrite": False,
    "max_files": None,
    "max_samples_per_file": None,
}

SCHEMA_VERSION = "4d-imagenet-disk-labels-v1"


def discover_merged_files(input_root: Path) -> list[dict[str, Any]]:
    """Select one acquisition-level merged PKL from each material/acquisition folder."""
    acquisitions: list[dict[str, Any]] = []
    for material_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        for acquisition_dir in sorted(path for path in material_dir.iterdir() if path.is_dir()):
            merged = acquisition_dir / "merged_5000.pkl"
            named = acquisition_dir / f"{acquisition_dir.name}.pkl"

            if merged.is_file():
                selected = merged
            elif named.is_file():
                selected = named
            else:
                large_candidates = sorted(
                    path
                    for path in acquisition_dir.glob("*.pkl")
                    if path.is_file() and path.stat().st_size > 500 * 1024 * 1024
                )
                if len(large_candidates) != 1:
                    raise RuntimeError(
                        f"Expected one acquisition-level merged PKL in {acquisition_dir}, "
                        f"found {len(large_candidates)} candidates."
                    )
                selected = large_candidates[0]

            acquisitions.append(
                {
                    "material": material_dir.name,
                    "acquisition": acquisition_dir.name,
                    "source_file": selected,
                }
            )

    if not acquisitions:
        raise RuntimeError(f"No acquisition directories were found under {input_root}")
    return acquisitions


def output_path_for(acquisition: dict[str, Any], output_root: Path) -> Path:
    return output_root / acquisition["material"] / f"{acquisition['acquisition']}.json"


def _preprocess_sample(sample: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    tensor, display_image = preprocess_pattern(
        sample["pattern"],
        int(config["image_size"]),
        str(config["preprocess"]),
    )
    sample["tensor"] = tensor
    sample["display_image"] = display_image
    return sample


def _predict_with_oom_fallback(
    model,
    batch: list[dict[str, Any]],
    config: dict[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    try:
        return predict_batch_records(model, batch, config, device)
    except torch.cuda.OutOfMemoryError:
        if device.type != "cuda" or len(batch) == 1:
            raise
        torch.cuda.empty_cache()
        midpoint = len(batch) // 2
        print(f"CUDA OOM at batch {len(batch)}; retrying as {midpoint} + {len(batch) - midpoint}.")
        return _predict_with_oom_fallback(model, batch[:midpoint], config, device) + _predict_with_oom_fallback(
            model, batch[midpoint:], config, device
        )


def _compact_prediction(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata") or {}
    return {
        "index": int(record["index"]),
        "scan_position": _jsonable(metadata.get("scan_position")),
        "original_shape": record["original_shape"],
        "center_beam": record["center_beam"],
        "disk_count": int(record["disk_count"]),
        "disks": record["disks"],
    }


def _relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def process_acquisition(
    model,
    acquisition: dict[str, Any],
    output_root: Path,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    source_file = Path(acquisition["source_file"])
    output_path = output_path_for(acquisition, output_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".json.tmp")

    if output_path.exists() and not bool(config["overwrite"]):
        return {
            "material": acquisition["material"],
            "acquisition": acquisition["acquisition"],
            "source_file": _relative_path(source_file),
            "label_file": _relative_path(output_path),
            "status": "skipped_existing",
        }

    temporary_path.unlink(missing_ok=True)
    samples = iter(_iter_pkl_patterns(source_file))
    try:
        first_sample = next(samples)
    except StopIteration as exc:
        raise RuntimeError(f"No diffraction patterns found in {source_file}") from exc

    sample_limit = config.get("max_samples_per_file")
    sample_limit = int(sample_limit) if sample_limit is not None else None
    sample_iterator: Iterable[dict[str, Any]] = chain([first_sample], samples)

    material = first_sample.get("metadata", {}).get("material") or acquisition["material"]
    estimated_beam_center = first_sample.get("metadata", {}).get("estimated_beam_center")
    weight_path = Path(config["weight"]).resolve() if config.get("weight") else WEIGHT_PATHS[config["version"]]

    header = {
        "schema_version": SCHEMA_VERSION,
        "label_type": "model_predicted_disk_centres",
        "source_file": _relative_path(source_file),
        "material": _jsonable(material),
        "acquisition": acquisition["acquisition"],
        "estimated_beam_center": _jsonable(estimated_beam_center),
        "model": {
            "version": config["version"],
            "weight": _relative_path(weight_path),
            "confidence_threshold": float(config["confidence"]),
            "preprocess": config["preprocess"],
            "image_size": int(config["image_size"]),
        },
    }

    processed_count = 0
    total_disks = 0
    minimum_disks: int | None = None
    maximum_disks = 0
    first_prediction = True
    pending: list[dict[str, Any]] = []

    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("{\n")
        for key, value in header.items():
            handle.write(f"  {json.dumps(key)}: {json.dumps(value, ensure_ascii=False, default=_jsonable)},\n")
        handle.write('  "predictions": [\n')

        progress = tqdm(
            total=sample_limit,
            desc=f"{acquisition['material']}/{acquisition['acquisition']}",
            unit="pattern",
        )
        try:
            for sample in sample_iterator:
                if sample_limit is not None and processed_count + len(pending) >= sample_limit:
                    break
                pending.append(_preprocess_sample(sample, config))
                if len(pending) < int(config["batch_size"]):
                    continue

                records = _predict_with_oom_fallback(model, pending, config, device)
                for record in records:
                    compact = _compact_prediction(record)
                    if not first_prediction:
                        handle.write(",\n")
                    handle.write("    " + json.dumps(compact, ensure_ascii=False, separators=(",", ":")))
                    first_prediction = False
                    disk_count = int(compact["disk_count"])
                    total_disks += disk_count
                    minimum_disks = disk_count if minimum_disks is None else min(minimum_disks, disk_count)
                    maximum_disks = max(maximum_disks, disk_count)
                processed_count += len(records)
                progress.update(len(records))
                pending = []

            if pending:
                records = _predict_with_oom_fallback(model, pending, config, device)
                for record in records:
                    compact = _compact_prediction(record)
                    if not first_prediction:
                        handle.write(",\n")
                    handle.write("    " + json.dumps(compact, ensure_ascii=False, separators=(",", ":")))
                    first_prediction = False
                    disk_count = int(compact["disk_count"])
                    total_disks += disk_count
                    minimum_disks = disk_count if minimum_disks is None else min(minimum_disks, disk_count)
                    maximum_disks = max(maximum_disks, disk_count)
                processed_count += len(records)
                progress.update(len(records))
        finally:
            progress.close()

        summary = {
            "pattern_count": processed_count,
            "total_disk_count": total_disks,
            "mean_disk_count": total_disks / processed_count if processed_count else 0.0,
            "minimum_disk_count": minimum_disks or 0,
            "maximum_disk_count": maximum_disks,
        }
        handle.write("\n  ],\n")
        handle.write(f'  "summary": {json.dumps(summary, ensure_ascii=False)}\n')
        handle.write("}\n")

    temporary_path.replace(output_path)
    result = {
        "material": acquisition["material"],
        "acquisition": acquisition["acquisition"],
        "source_file": _relative_path(source_file),
        "label_file": _relative_path(output_path),
        "status": "completed",
        **summary,
    }
    print(
        f"Saved {processed_count} patterns and {total_disks} disks to "
        f"{output_path.relative_to(PROJECT_ROOT)}"
    )
    return result


def write_manifest(output_root: Path, config: dict[str, Any], entries: list[dict[str, Any]]) -> Path:
    completed = [entry for entry in entries if entry["status"] == "completed"]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "model_version": config["version"],
        "confidence_threshold": float(config["confidence"]),
        "acquisition_count": len(entries),
        "completed_in_this_run": len(completed),
        "pattern_count_in_this_run": sum(int(entry.get("pattern_count", 0)) for entry in completed),
        "entries": entries,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def run(config: dict[str, Any]) -> Path:
    input_root = Path(config["input_root"]).resolve()
    output_root = Path(config["output_root"]).resolve()
    acquisitions = discover_merged_files(input_root)
    max_files = config.get("max_files")
    if max_files is not None:
        acquisitions = acquisitions[: int(max_files)]

    print(f"Selected {len(acquisitions)} acquisition-level merged PKL files from {input_root}")
    device = torch.device(config["device"])
    weight_path = Path(config["weight"]).resolve() if config.get("weight") else WEIGHT_PATHS[config["version"]]
    model = build_model(config["version"], weight_path, device)

    entries: list[dict[str, Any]] = []
    for index, acquisition in enumerate(acquisitions, start=1):
        print(f"[{index}/{len(acquisitions)}] {acquisition['material']}/{acquisition['acquisition']}")
        entries.append(process_acquisition(model, acquisition, output_root, config, device))
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    manifest_path = write_manifest(output_root, config, entries)
    print(f"Saved label manifest to {manifest_path}")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one disk-label JSON for each experimental acquisition.")
    parser.add_argument("--input-root", type=Path, default=RUN_CONFIG["input_root"])
    parser.add_argument("--output-root", type=Path, default=RUN_CONFIG["output_root"])
    parser.add_argument("--version", choices=sorted(MODEL_CONFIGS), default=RUN_CONFIG["version"])
    parser.add_argument("--weight", type=Path, default=RUN_CONFIG["weight"])
    parser.add_argument("--device", default=RUN_CONFIG["device"])
    parser.add_argument("--confidence", type=float, default=RUN_CONFIG["confidence"])
    parser.add_argument("--batch-size", type=int, default=RUN_CONFIG["batch_size"])
    parser.add_argument("--max-files", type=int, default=RUN_CONFIG["max_files"])
    parser.add_argument("--max-samples-per-file", type=int, default=RUN_CONFIG["max_samples_per_file"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = dict(RUN_CONFIG)
    config.update(
        {
            "input_root": args.input_root,
            "output_root": args.output_root,
            "version": args.version,
            "weight": args.weight,
            "device": args.device,
            "confidence": args.confidence,
            "batch_size": args.batch_size,
            "max_files": args.max_files,
            "max_samples_per_file": args.max_samples_per_file,
            "overwrite": args.overwrite,
        }
    )
    run(config)
