import argparse
import importlib
import pickle
import re
from pathlib import Path

import numpy as np


SAMPLE_PKL_RE = re.compile(r"^\d{4}_x\d+_y\d+\.pkl$")


class NumpyCompatUnpickler(pickle.Unpickler):
    MODULE_CANDIDATES = {
        ("numpy._core.numeric", "_frombuffer"): [
            ("numpy._core.numeric", "_frombuffer"),
            ("numpy.core.numeric", "_frombuffer"),
        ],
        ("numpy._core.multiarray", "_reconstruct"): [
            ("numpy._core.multiarray", "_reconstruct"),
            ("numpy.core.multiarray", "_reconstruct"),
        ],
        ("numpy.core.numeric", "_frombuffer"): [
            ("numpy._core.numeric", "_frombuffer"),
            ("numpy.core.numeric", "_frombuffer"),
        ],
        ("numpy.core.multiarray", "_reconstruct"): [
            ("numpy._core.multiarray", "_reconstruct"),
            ("numpy.core.multiarray", "_reconstruct"),
        ],
    }

    def find_class(self, module, name):
        candidates = self.MODULE_CANDIDATES.get((module, name))
        if candidates is not None:
            for candidate_module, candidate_name in candidates:
                try:
                    imported = importlib.import_module(candidate_module)
                    getattr(imported, candidate_name)
                    module, name = candidate_module, candidate_name
                    break
                except (ImportError, AttributeError):
                    continue
        return super().find_class(module, name)


def load_payload(path):
    with open(path, "rb") as f:
        return NumpyCompatUnpickler(f).load()


def list_sample_pkls(folder):
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and SAMPLE_PKL_RE.fullmatch(path.name)
    )


def collect_selection_metrics(payloads):
    metrics = {
        "center_fraction": np.array(
            [payload["selection_metrics"]["center_fraction"] for payload in payloads],
            dtype=np.float32,
        ),
        "outer_fraction": np.array(
            [payload["selection_metrics"]["outer_fraction"] for payload in payloads],
            dtype=np.float32,
        ),
        "non_center_std": np.array(
            [payload["selection_metrics"]["non_center_std"] for payload in payloads],
            dtype=np.float32,
        ),
        "non_center_top1_fraction": np.array(
            [payload["selection_metrics"]["non_center_top1_fraction"] for payload in payloads],
            dtype=np.float32,
        ),
        "beam_center_distance": np.array(
            [payload["selection_metrics"]["beam_center_distance"] for payload in payloads],
            dtype=np.float32,
        ),
        "center_of_mass": np.array(
            [
                [
                    payload["selection_metrics"]["center_of_mass"]["y"],
                    payload["selection_metrics"]["center_of_mass"]["x"],
                ]
                for payload in payloads
            ],
            dtype=np.float32,
        ),
    }
    return metrics


def merge_folder(folder, output_name=None, overwrite=False):
    sample_files = list_sample_pkls(folder)
    if not sample_files:
        raise FileNotFoundError(f"No sample pkl files found in {folder}")

    if output_name is None:
        output_name = f"{folder.name}.pkl"
    output_path = folder / output_name
    if output_path.exists() and not overwrite:
        return {
            "folder": str(folder),
            "output_path": str(output_path),
            "sample_count": len(sample_files),
            "skipped_existing": True,
        }

    payloads = [load_payload(path) for path in sample_files]
    first = payloads[0]
    diffraction_shape = tuple(first["diffraction_pattern"].shape)

    patterns = np.empty((len(payloads),) + diffraction_shape, dtype=np.float32)
    scan_positions = np.empty((len(payloads), 2), dtype=np.int32)
    selection_ranks = np.empty(len(payloads), dtype=np.int32)
    scores = np.empty(len(payloads), dtype=np.float32)

    for idx, payload in enumerate(payloads):
        pattern = np.asarray(payload["diffraction_pattern"], dtype=np.float32)
        if tuple(pattern.shape) != diffraction_shape:
            raise ValueError(f"Inconsistent diffraction shape in {folder}: {pattern.shape} != {diffraction_shape}")
        patterns[idx] = pattern
        scan_positions[idx, 0] = int(payload["scan_position"]["x"])
        scan_positions[idx, 1] = int(payload["scan_position"]["y"])
        selection_ranks[idx] = int(payload["selection_rank"])
        scores[idx] = float(payload["score"])

    merged = {
        "source_file": first["source_file"],
        "source_name": first["source_name"],
        "material": first["material"],
        "count": len(payloads),
        "scan_shape": first["scan_shape"],
        "diffraction_shape": first["diffraction_shape"],
        "fov": first["fov"],
        "camera_length": first["camera_length"],
        "real_space_unit": first["real_space_unit"],
        "filename_scan_shape": first["filename_scan_shape"],
        "estimated_beam_center": first.get("estimated_beam_center"),
        "scan_positions": scan_positions,
        "selection_ranks": selection_ranks,
        "scores": scores,
        "diffraction_patterns": patterns,
        "selection_metrics": collect_selection_metrics(payloads),
        "sample_files": [path.name for path in sample_files],
    }

    with open(output_path, "wb") as f:
        pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)

    return {
        "folder": str(folder),
        "output_path": str(output_path),
        "sample_count": len(payloads),
        "skipped_existing": False,
    }


def find_pattern_dirs(src_root):
    src_root = Path(src_root)
    if src_root.is_dir():
        direct_pkls = list_sample_pkls(src_root)
        if direct_pkls:
            return [src_root]
        return sorted(path for path in src_root.rglob("*_master") if path.is_dir() and list_sample_pkls(path))
    raise FileNotFoundError(f"Path not found or not a directory: {src_root}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Merge extracted diffraction pkl files")
    parser.add_argument(
        "--src-root",
        default=".",
        help="Workspace root, material directory, or a single *_master directory.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Merged pkl filename written inside each *_master directory. Defaults to <folder_name>.pkl",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing merged pkl file.",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    pattern_dirs = find_pattern_dirs(args.src_root)
    if not pattern_dirs:
        raise FileNotFoundError(f"No extracted pkl directories found under {args.src_root}")

    for folder in pattern_dirs:
        print(f"[merge] {folder}")
        summary = merge_folder(folder, output_name=args.output_name, overwrite=args.overwrite)
        if summary["skipped_existing"]:
            print(f"[skip] {summary['output_path']} already exists")
        else:
            print(f"[done] {summary['sample_count']} -> {summary['output_path']}")


if __name__ == "__main__":
    main()
