from __future__ import annotations

import argparse
import gc
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def canonical_experimental_files(root: Path) -> list[Path]:
    files = []
    for folder in sorted(path for path in root.rglob("*_master") if path.is_dir()):
        canonical = folder / f"{folder.name}.pkl"
        fallback = folder / "merged_5000.pkl"
        if canonical.exists():
            files.append(canonical)
        elif fallback.exists():
            files.append(fallback)
    return files


def audit_experimental(path: Path, sample_count: int) -> dict:
    with path.open("rb") as handle:
        payload = pickle.load(handle)

    required = {
        "source_file",
        "source_name",
        "material",
        "count",
        "scan_shape",
        "diffraction_shape",
        "scan_positions",
        "selection_ranks",
        "scores",
        "diffraction_patterns",
        "selection_metrics",
    }
    missing = sorted(required - set(payload))
    patterns = np.asarray(payload["diffraction_patterns"])
    positions = np.asarray(payload["scan_positions"])
    ranks = np.asarray(payload["selection_ranks"])
    scores = np.asarray(payload["scores"])
    count = int(payload["count"])
    sx = int(payload["scan_shape"]["x"])
    sy = int(payload["scan_shape"]["y"])

    indices = np.linspace(0, count - 1, min(sample_count, count), dtype=int)
    sampled_patterns = patterns[indices]
    metric_lengths = {
        key: int(np.asarray(value).shape[0])
        for key, value in payload["selection_metrics"].items()
    }
    checks = {
        "required_fields": not missing,
        "count_matches_patterns": patterns.shape[0] == count,
        "count_matches_positions": positions.shape == (count, 2),
        "count_matches_ranks": ranks.shape == (count,),
        "count_matches_scores": scores.shape == (count,),
        "metric_lengths_match": all(length == count for length in metric_lengths.values()),
        "positions_in_bounds": bool(
            np.all((positions[:, 0] >= 0) & (positions[:, 0] < sx))
            and np.all((positions[:, 1] >= 0) & (positions[:, 1] < sy))
        ),
        "ranks_are_sequential": bool(np.array_equal(ranks, np.arange(count))),
        "scores_nonincreasing": bool(np.all(scores[:-1] >= scores[1:])),
        "sampled_values_finite": bool(np.isfinite(sampled_patterns).all()),
        "sampled_values_nonnegative": bool((sampled_patterns >= 0).all()),
    }
    result = {
        "path": str(path),
        "material": payload["material"],
        "count": count,
        "scan_shape": [sx, sy],
        "diffraction_shape": list(patterns.shape[1:]),
        "dtype": str(patterns.dtype),
        "missing_fields": missing,
        "checks": checks,
        "passed": all(checks.values()),
    }
    del payload, patterns, positions, ranks, scores, sampled_patterns
    gc.collect()
    return result


def audit_simulation(path: Path, sample_count: int) -> dict:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    patterns = np.asarray(payload["diffraction_patterns"])
    count = int(payload["count"])
    indices = np.linspace(0, count - 1, min(sample_count, count), dtype=int)
    sampled_patterns = patterns[indices]
    aligned_fields = ["center_list", "pixel_size_list", "orientation_list", "hkl_list"]
    checks = {
        "count_matches_patterns": patterns.shape[0] == count,
        "aligned_metadata_lengths": all(np.asarray(payload[key]).shape[0] == count for key in aligned_fields),
        "sampled_values_finite": bool(np.isfinite(sampled_patterns).all()),
        "sampled_values_nonnegative": bool((sampled_patterns >= 0).all()),
        "sampled_patterns_normalized": bool(
            np.allclose(sampled_patterns.sum(axis=(1, 2)), 1.0, rtol=1e-3, atol=1e-5)
        ),
    }
    result = {
        "path": str(path),
        "material": payload["material"],
        "count": count,
        "diffraction_shape": list(patterns.shape[1:]),
        "dtype": str(patterns.dtype),
        "checks": checks,
        "passed": all(checks.values()),
    }
    del payload, patterns, sampled_patterns
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit canonical 4D-ImageNet merged PKL files")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "manuscript" / "dataset_audit.json",
    )
    args = parser.parse_args()

    experimental_files = canonical_experimental_files(args.root / "exp")
    simulation_files = sorted((args.root / "sim").rglob("*.pkl"))
    experimental = []
    simulation = []

    for index, path in enumerate(experimental_files, start=1):
        print(f"[experimental {index}/{len(experimental_files)}] {path.name}", flush=True)
        experimental.append(audit_experimental(path, args.sample_count))
    for index, path in enumerate(simulation_files, start=1):
        print(f"[simulation {index}/{len(simulation_files)}] {path.name}", flush=True)
        simulation.append(audit_simulation(path, args.sample_count))

    summary = {
        "experimental_acquisitions": len(experimental),
        "experimental_patterns": sum(item["count"] for item in experimental),
        "experimental_materials": dict(Counter(item["material"] for item in experimental)),
        "experimental_passed": sum(item["passed"] for item in experimental),
        "simulation_files": len(simulation),
        "simulation_patterns": sum(item["count"] for item in simulation),
        "simulation_materials": dict(Counter(item["material"] for item in simulation)),
        "simulation_passed": sum(item["passed"] for item in simulation),
        "total_patterns": sum(item["count"] for item in experimental + simulation),
    }
    report = {"summary": summary, "experimental": experimental, "simulation": simulation}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved audit report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
