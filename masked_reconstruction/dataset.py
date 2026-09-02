import csv
import importlib
import pickle
import random
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


PATH_COLUMNS = ("path", "file", "filepath", "file_path", "pkl_path", "filename")
METADATA_COLUMNS = ("metadata_id", "metadata", "acquisition_id", "source_name", "source_file")


def _print_progress(label: str, current: int, total: int, detail: str = ""):
    total = max(int(total), 1)
    current = min(max(int(current), 0), total)
    width = 28
    filled = int(round(width * current / total))
    bar = "#" * filled + "-" * (width - filled)
    suffix = f" {detail}" if detail else ""
    print(f"\r{label} [{bar}] {current}/{total}{suffix}", end="", flush=True)
    if current >= total:
        print(flush=True)


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


def load_pkl(path: Path) -> dict:
    with open(path, "rb") as f:
        obj = NumpyCompatUnpickler(f).load()
    if not isinstance(obj, dict):
        raise ValueError(f"{path} does not contain a dictionary payload")
    return obj


def _read_csv(path: Path) -> List[dict]:
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _resolve_path(value: str, data_root: Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return data_root / p


def _first_existing_column(row: dict, columns: Sequence[str]) -> Optional[str]:
    for col in columns:
        value = row.get(col)
        if value not in (None, ""):
            return value
    return None


def discover_pkl_files(
    data_root: Path,
    file_index: Optional[Path] = None,
    max_files: Optional[int] = None,
    file_pattern: str = "*.pkl",
) -> List[Path]:
    data_root = Path(data_root)
    if file_index and Path(file_index).exists():
        rows = _read_csv(Path(file_index))
        files = []
        for row in rows:
            value = _first_existing_column(row, PATH_COLUMNS)
            if value is None:
                continue
            p = _resolve_path(value, data_root)
            if p.suffix.lower() == ".pkl" and p.exists():
                files.append(p)
        if not files:
            raise ValueError(f"No readable PKL files found from file index: {file_index}")
    else:
        files = sorted(data_root.rglob(file_pattern))

    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No PKL files found under {data_root}")
    return files


def load_metadata_groups(metadata_path: Optional[Path], file_index: Optional[Path], data_root: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for csv_path in (file_index, metadata_path):
        if not csv_path or not Path(csv_path).exists():
            continue
        for row in _read_csv(Path(csv_path)):
            path_value = _first_existing_column(row, PATH_COLUMNS)
            group_value = _first_existing_column(row, METADATA_COLUMNS)
            if path_value is None or group_value is None:
                continue
            resolved = str(_resolve_path(path_value, data_root).resolve())
            mapping[resolved] = str(group_value)
    return mapping


def inspect_pkl_samples(path: Path, metadata_id: str) -> List[dict]:
    payload = load_pkl(path)
    if "diffraction_pattern" in payload:
        return [{"path": path, "index": None, "metadata_id": metadata_id}]
    if "diffraction_patterns" in payload:
        patterns = payload["diffraction_patterns"]
        if not hasattr(patterns, "shape") or len(patterns.shape) != 3:
            raise ValueError(f"{path} has 'diffraction_patterns' but expected shape (N,H,W), got {getattr(patterns, 'shape', None)}")
        return [{"path": path, "index": i, "metadata_id": metadata_id} for i in range(int(patterns.shape[0]))]
    raise KeyError(f"{path} must contain 'diffraction_pattern' or 'diffraction_patterns'")


def _select_evenly(items: List[dict], count: int) -> List[dict]:
    if count <= 0:
        return []
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[0]]
    last = len(items) - 1
    indices = [round(i * last / (count - 1)) for i in range(count)]
    return [items[i] for i in indices]


def build_sample_index(
    data_root: Path,
    metadata_path: Optional[Path] = None,
    file_index: Optional[Path] = None,
    max_files: Optional[int] = None,
    max_samples: Optional[int] = None,
    file_pattern: str = "*.pkl",
    show_progress: bool = True,
) -> Tuple[List[dict], bool]:
    files = discover_pkl_files(
        data_root,
        file_index=file_index,
        max_files=max_files,
        file_pattern=file_pattern,
    )
    metadata_map = load_metadata_groups(metadata_path, file_index, data_root)
    has_external_metadata_groups = bool(metadata_map)
    per_file_samples: List[List[dict]] = []
    if show_progress:
        print(f"Found {len(files)} PKL file(s) under {data_root}")
    for i, path in enumerate(files, start=1):
        if show_progress:
            _print_progress("Indexing PKL files", i, len(files), path.parent.name)
        resolved = str(path.resolve())
        # If metadata.csv/file_index.csv is unavailable, use the PKL path itself as
        # a grouping id. This prevents samples from the same merged acquisition file
        # leaking across train/val/test splits.
        metadata_id = metadata_map.get(resolved, resolved)
        per_file_samples.append(inspect_pkl_samples(path, metadata_id))

    if max_samples is None:
        samples = [sample for group in per_file_samples for sample in group]
        if show_progress:
            print(f"Indexed {len(samples)} diffraction pattern(s)")
        return samples, True

    max_samples = max(0, int(max_samples))
    if max_samples == 0:
        return [], True

    total_available = sum(len(group) for group in per_file_samples)
    if total_available <= max_samples:
        samples = [sample for group in per_file_samples for sample in group]
        if show_progress:
            print(f"Indexed {len(samples)} diffraction pattern(s)")
        return samples, True

    # When data_root contains multiple merged PKLs, sample across files instead of
    # filling max_samples from the first file only. This keeps quick debug runs
    # representative of all materials under the root.
    non_empty = [group for group in per_file_samples if group]
    file_count = len(non_empty)
    base = max_samples // file_count
    remainder = max_samples % file_count
    selected: List[dict] = []
    for i, group in enumerate(non_empty):
        quota = base + (1 if i < remainder else 0)
        selected.extend(_select_evenly(group, quota))
    selected = selected[:max_samples]
    if show_progress:
        print(f"Selected {len(selected)} / {total_available} diffraction pattern(s) across {file_count} PKL file(s)")
        if not has_external_metadata_groups:
            print("No metadata_id column found; using each PKL file path as the split group.")
    return selected, True


def split_samples(
    samples: List[dict],
    ratios: Tuple[float, float, float],
    seed: int,
    use_metadata_groups: bool,
) -> Tuple[List[dict], List[dict], List[dict]]:
    train_ratio, val_ratio, test_ratio = ratios
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value")
    train_ratio, val_ratio, test_ratio = train_ratio / total, val_ratio / total, test_ratio / total

    rng = random.Random(seed)
    if use_metadata_groups:
        groups: Dict[str, List[dict]] = {}
        for sample in samples:
            groups.setdefault(sample["metadata_id"], []).append(sample)
        group_items = list(groups.items())
        rng.shuffle(group_items)
        n_groups = len(group_items)
        n_train = int(round(train_ratio * n_groups))
        n_val = int(round(val_ratio * n_groups))
        train_groups = group_items[:n_train]
        val_groups = group_items[n_train:n_train + n_val]
        test_groups = group_items[n_train + n_val:]
        return (
            [s for _, group in train_groups for s in group],
            [s for _, group in val_groups for s in group],
            [s for _, group in test_groups for s in group],
        )

    warnings.warn("metadata_id was not available; using fallback random sample-level split.", RuntimeWarning)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(train_ratio * n))
    n_val = int(round(val_ratio * n))
    return shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]


class DiffractionPatternDataset(Dataset):
    def __init__(self, samples: Sequence[dict], image_size: Optional[int] = None, cache_size: int = 2, preload_limit: int = 2000):
        self.samples = list(samples)
        self.image_size = image_size
        self.cache_size = max(1, int(cache_size))
        self._cache: OrderedDict[Path, dict] = OrderedDict()
        self._preloaded_images: Optional[List[torch.Tensor]] = None
        preload_limit = int(preload_limit)
        if 0 < preload_limit and len(self.samples) <= preload_limit:
            self._preload_selected_samples()
        elif preload_limit > 0:
            print(f"Not preloading {len(self.samples)} sample(s): split size exceeds preload_limit={preload_limit}")
        else:
            print(f"Preloading disabled for {len(self.samples)} sample(s)")

    def _preload_selected_samples(self):
        self._preloaded_images = [None] * len(self.samples)  # type: ignore[list-item]
        grouped: Dict[Path, List[Tuple[int, dict]]] = {}
        for sample_idx, sample in enumerate(self.samples):
            grouped.setdefault(Path(sample["path"]), []).append((sample_idx, sample))
        total_entries = sum(len(entries) for entries in grouped.values())
        loaded_entries = 0
        print(f"Preloading {total_entries} selected sample(s) from {len(grouped)} PKL file(s)")
        for path, entries in grouped.items():
            payload = load_pkl(path)
            for sample_idx, sample in entries:
                if sample["index"] is None:
                    if "diffraction_pattern" not in payload:
                        raise KeyError(f"{path} does not contain 'diffraction_pattern'")
                    pattern = np.asarray(payload["diffraction_pattern"])
                else:
                    if "diffraction_patterns" not in payload:
                        raise KeyError(f"{path} does not contain 'diffraction_patterns'")
                    pattern = np.asarray(payload["diffraction_patterns"][sample["index"]])
                tensor = self.preprocess(pattern)
                self._preloaded_images[sample_idx] = self._resize_if_needed(tensor)  # type: ignore[index]
                loaded_entries += 1
                _print_progress("Preloading samples", loaded_entries, total_entries, path.parent.name)

    def __len__(self) -> int:
        return len(self.samples)

    def _payload(self, path: Path) -> dict:
        path = Path(path)
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        payload = load_pkl(path)
        self._cache[path] = payload
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return payload

    def _pattern_from_sample(self, sample: dict) -> np.ndarray:
        payload = self._payload(sample["path"])
        if sample["index"] is None:
            if "diffraction_pattern" not in payload:
                raise KeyError(f"{sample['path']} does not contain 'diffraction_pattern'")
            return np.asarray(payload["diffraction_pattern"])
        if "diffraction_patterns" not in payload:
            raise KeyError(f"{sample['path']} does not contain 'diffraction_patterns'")
        return np.asarray(payload["diffraction_patterns"][sample["index"]])

    @staticmethod
    def preprocess(pattern: np.ndarray) -> torch.Tensor:
        arr = np.asarray(pattern, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr[arr < 0] = 0.0
        arr = np.log1p(arr)
        max_value = float(arr.max())
        if max_value > 0:
            arr = arr / (max_value + 1e-8)
        return torch.from_numpy(arr).unsqueeze(0).float()

    def _resize_if_needed(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.image_size is None:
            return tensor
        if tensor.shape[-1] == self.image_size and tensor.shape[-2] == self.image_size:
            return tensor
        x = tensor.unsqueeze(0)
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return x.squeeze(0)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        if self._preloaded_images is not None:
            pattern = self._preloaded_images[idx]
        else:
            pattern = self.preprocess(self._pattern_from_sample(sample))
            pattern = self._resize_if_needed(pattern)
        return {"image": pattern, "path": str(sample["path"]), "index": -1 if sample["index"] is None else int(sample["index"]), "metadata_id": sample["metadata_id"]}
